#!/usr/bin/env python3
import os
import time
import random
import subprocess

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from picarx import Picarx

# ================== MODEL / CAMERA ==================
MODEL_PATH = "/home/pi/Desktop/AFV_FIRE_AI/runs/yolov8n-fire2/weights/last.pt"
WIDTH, HEIGHT, FPS = 640, 480, 5  # lower FPS to match YOLO speed

RPICAM_CMD = [
    "rpicam-vid",
    "--nopreview",
    "--timeout", "0",
    "--low-latency",          # reduce encoder buffering
    "--width", str(WIDTH),
    "--height", str(HEIGHT),
    "--framerate", str(FPS),
    "--codec", "mjpeg",
    "-o", "-"
]

IMGSZ       = 480
CONF_THRESH = 0.5
WINDOW_NAME = "AFV: Slow Wander + Low-Gain Cam Track"

# ================== OBSTACLE AVOID (slow like avoiding_obstacles.py) ==================
SAFE_CM    = 40
DANGER_CM  = 20

# VERY gentle wheel speeds because of camera latency
CRUISE_PWR     = 5   # safe forward
CAUTION_PWR    = 5    # forward in caution nudge
BACK_PWR       = 5    # backward in danger
CAUTION_STEER  = 10   # deg
HARD_TURN      = 10   # deg
BACKUP_S       = 0.7
CAUTION_HOLD_S = 0.3

# ================== TRACK (camera pan/tilt centering) ==================
CAM_LIMIT = 35.0           # ±deg
TRACK_BASE_SPEED = 8       # very slow creep
TRACK_MAX_SPEED  = 12
LOCK_TIMEOUT_S   = 2.0     # keep lock a bit longer

# max degrees per frame (5 fps -> ~5 deg/s)
CAM_MAX_STEP = 1.0


def clamp(num, lo, hi):
    return max(lo, min(hi, num))


# ================== MJPEG PARSER ==================
def mjpeg_frames(stream):
    buf = b""
    SOI = b"\xff\xd8"
    EOI = b"\xff\xd9"
    while True:
        chunk = stream.read(4096)
        if not chunk:
            break
        buf += chunk
        while True:
            s = buf.find(SOI)
            if s == -1:
                break
            e = buf.find(EOI, s + 2)
            if e == -1:
                break
            yield buf[s:e + 2]
            buf = buf[e + 2:]


# ================== CAR WRAPPER ==================
class Car:
    def __init__(self):
        self.px = Picarx()

        # camera API from your stare_at_you.py
        if not hasattr(self.px, "set_cam_pan_angle") or not hasattr(self.px, "set_cam_tilt_angle"):
            print("[WARN] Picarx camera pan/tilt functions not found!")

        self.cam_pan = 0.0
        self.cam_tilt = 0.0

        self.center_all()
        self.stop()

    # drive
    def forward(self, pwr):
        self.px.forward(int(clamp(pwr, 0, 100)))

    def backward(self, pwr):
        self.px.backward(int(clamp(pwr, 0, 100)))

    def stop(self):
        self.px.stop()

    def steer(self, deg):
        deg = float(clamp(deg, -35, 35))
        self.px.set_dir_servo_angle(deg)

    def distance_cm(self):
        try:
            d = self.px.ultrasonic.read()
            return float(d) if d and d > 0 else 999.0
        except Exception:
            return 999.0

    # camera servos
    def set_cam_angles(self, pan_deg, tilt_deg):
        self.cam_pan = clamp(pan_deg, -CAM_LIMIT, CAM_LIMIT)
        self.cam_tilt = clamp(tilt_deg, -CAM_LIMIT, CAM_LIMIT)
        try:
            self.px.set_cam_pan_angle(self.cam_pan)
            self.px.set_cam_tilt_angle(self.cam_tilt)
        except Exception as e:
            print("[WARN] set_cam_* failed:", e)

    def nudge_cam_slow(self, cx, cy):
        """
        Low-gain pan/tilt:
        - normalized error [-1,1] in x & y
        - per-frame step is clamped to ±CAM_MAX_STEP deg
        """
        ex = (cx - WIDTH / 2.0) / (WIDTH / 2.0)   # -1 .. 1
        ey = (cy - HEIGHT / 2.0) / (HEIGHT / 2.0)

        pan_step = clamp(ex * CAM_MAX_STEP, -CAM_MAX_STEP, CAM_MAX_STEP)
        tilt_step = clamp(-ey * CAM_MAX_STEP, -CAM_MAX_STEP, CAM_MAX_STEP)

        self.set_cam_angles(self.cam_pan + pan_step,
                            self.cam_tilt + tilt_step)

    def center_cam(self):
        self.set_cam_angles(0.0, 0.0)

    def center_all(self):
        self.center_cam()
        self.steer(0.0)


# ================== FIRE LOCK STATE ==================
class FireLock:
    def __init__(self):
        self.ts = 0.0
        self.cx = self.cy = None
        self.area = 0.0
        self.conf = 0.0

    def update(self, cx, cy, area, conf):
        self.ts = time.time()
        self.cx = cx
        self.cy = cy
        self.area = area
        self.conf = conf

    def mode_target(self):
        age = time.time() - self.ts
        if age <= LOCK_TIMEOUT_S and self.conf >= CONF_THRESH:
            return "TRACK", self.cx, self.cy, self.area
        return "WANDER", None, None, 0.0


# ================== MAIN ==================
def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    model = YOLO(MODEL_PATH).to(device)

    car = Car()
    lock = FireLock()

    # Headless if there is no DISPLAY, or explicitly offscreen
    HEADLESS = not os.environ.get("DISPLAY") or os.environ.get("QT_QPA_PLATFORM") == "offscreen"
    if HEADLESS:
        print("[INFO] Running in HEADLESS mode (no OpenCV window).")

    print("[INFO] starting rpicam-vid (MJPEG, low-latency) ...")
    proc = subprocess.Popen(
        RPICAM_CMD,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0
    )

    if not HEADLESS:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # Wander FSM
    state = "CRUISE"  # CRUISE, CAUTION, DANGER_BACK
    state_until = 0.0
    caution_dir = +1
    prev_mode = "WANDER"

    try:
        for jpeg in mjpeg_frames(proc.stdout):
            frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # -------- YOLO --------
            r = model.predict(
                frame, imgsz=IMGSZ, conf=CONF_THRESH, device=device, verbose=False
            )[0]

            best = None
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    conf = float(b.conf[0].item()) if b.conf is not None else 0.0
                    if conf < CONF_THRESH:
                        continue
                    w = max(1.0, x2 - x1)
                    h = max(1.0, y2 - y1)
                    area = w * h
                    cx = x1 + w / 2.0
                    cy = y1 + h / 2.0
                    if best is None or area > best[-2]:
                        best = (cx, cy, area, conf, (int(x1), int(y1), int(x2), int(y2)))

            if best:
                cx, cy, area, conf, bb = best
                lock.update(cx, cy, area, conf)
                x1, y1, x2, y2 = bb
                if not HEADLESS:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame, f"FIRE {conf:.2f}",
                        (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2, cv2.LINE_AA
                    )

            mode, cx, cy, area = lock.mode_target()

            # recenter gimbal when leaving TRACK
            if prev_mode == "TRACK" and mode == "WANDER":
                car.center_cam()

            if mode == "TRACK" and cx is not None and cy is not None:
                # -------- TRACK: slow pan/tilt follow --------
                car.nudge_cam_slow(cx, cy)

                # Only creep forward if roughly centered (small error)
                err_x = abs(cx - WIDTH / 2.0)
                err_y = abs(cy - HEIGHT / 2.0)
                centered = (err_x < WIDTH * 0.18) and (err_y < HEIGHT * 0.18)

                norm_area = area / float(WIDTH * HEIGHT)
                raw_speed = TRACK_BASE_SPEED + (TRACK_MAX_SPEED - TRACK_BASE_SPEED) * (
                    1.0 - min(1.0, 8.0 * norm_area)
                )
                speed = int(clamp(raw_speed, TRACK_BASE_SPEED, TRACK_MAX_SPEED))

                d = car.distance_cm()
                if centered and d > DANGER_CM:
                    car.steer(0.0)
                    car.forward(speed)
                else:
                    car.stop()

                if not HEADLESS:
                    cv2.putText(
                        frame,
                        f"MODE TRACK | pan={car.cam_pan:.1f} tilt={car.cam_tilt:.1f} spd={speed} d={int(d)}cm",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (40, 220, 40), 2, cv2.LINE_AA
                    )

            else:
                # -------- WANDER: slow safe/caution/danger FSM --------
                d = car.distance_cm()
                now = time.time()

                if state == "CRUISE":
                    if d >= SAFE_CM:
                        car.steer(0.0)
                        car.forward(CRUISE_PWR)
                    elif d >= DANGER_CM:
                        caution_dir = random.choice([+1, -1])
                        car.steer(caution_dir * CAUTION_STEER)
                        car.forward(CAUTION_PWR)
                        state = "CAUTION"
                        state_until = now + CAUTION_HOLD_S
                    else:
                        car.steer(-caution_dir * HARD_TURN)
                        car.backward(BACK_PWR)
                        state = "DANGER_BACK"
                        state_until = now + BACKUP_S

                elif state == "CAUTION":
                    if now >= state_until:
                        state = "CRUISE"
                    else:
                        if d < DANGER_CM:
                            car.steer(-caution_dir * HARD_TURN)
                            car.backward(BACK_PWR)
                            state = "DANGER_BACK"
                            state_until = now + BACKUP_S

                elif state == "DANGER_BACK":
                    if now >= state_until:
                        car.stop()
                        car.steer(-caution_dir * HARD_TURN)
                        car.forward(CAUTION_PWR)
                        time.sleep(0.25)
                        state = "CRUISE"

                if not HEADLESS:
                    cv2.putText(
                        frame,
                        f"MODE WANDER | state={state} d={int(d)}cm",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200, 200, 60), 2, cv2.LINE_AA
                    )

            prev_mode = mode

            # crosshair & display (only if GUI is available)
            if not HEADLESS:
                cv2.drawMarker(
                    frame,
                    (WIDTH // 2, HEIGHT // 2),
                    (255, 255, 255),
                    cv2.MARKER_CROSS,
                    20,
                    2
                )
                cv2.imshow(WINDOW_NAME, frame)
                if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
                    break
            else:
                # tiny sleep to avoid pegging CPU
                time.sleep(0.01)

    finally:
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            car.stop()
            car.center_all()
        except Exception:
            pass
        if not HEADLESS:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import os
import time
import random
import subprocess

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from picarx import Picarx

# ================== MODEL / CAMERA ==================
MODEL_PATH = "/home/pi/Desktop/AFV_FIRE_AI/runs/yolov8n-fire2/weights/last.pt"
WIDTH, HEIGHT, FPS = 640, 480, 5  # lower FPS to match YOLO speed

RPICAM_CMD = [
    "rpicam-vid",
    "--nopreview",
    "--timeout", "0",
    "--low-latency",          # reduce encoder buffering
    "--width", str(WIDTH),
    "--height", str(HEIGHT),
    "--framerate", str(FPS),
    "--codec", "mjpeg",
    "-o", "-"
]

IMGSZ       = 480
CONF_THRESH = 0.5
WINDOW_NAME = "AFV: Slow Wander + Low-Gain Cam Track"

# ================== OBSTACLE AVOID (slow like avoiding_obstacles.py) ==================
SAFE_CM    = 40
DANGER_CM  = 20

# VERY gentle wheel speeds because of camera latency
CRUISE_PWR     = 10   # safe forward
CAUTION_PWR    = 9    # forward in caution nudge
BACK_PWR       = 9    # backward in danger
CAUTION_STEER  = 25   # deg
HARD_TURN      = 35   # deg
BACKUP_S       = 0.7
CAUTION_HOLD_S = 0.3

# ================== TRACK (camera pan/tilt centering) ==================
CAM_LIMIT = 35.0           # ±deg
TRACK_BASE_SPEED = 8       # very slow creep
TRACK_MAX_SPEED  = 12
LOCK_TIMEOUT_S   = 2.0     # keep lock a bit longer

# max degrees per frame (5 fps -> ~5 deg/s)
CAM_MAX_STEP = 1.0


def clamp(num, lo, hi):
    return max(lo, min(hi, num))


# ================== MJPEG PARSER ==================
def mjpeg_frames(stream):
    buf = b""
    SOI = b"\xff\xd8"
    EOI = b"\xff\xd9"
    while True:
        chunk = stream.read(4096)
        if not chunk:
            break
        buf += chunk
        while True:
            s = buf.find(SOI)
            if s == -1:
                break
            e = buf.find(EOI, s + 2)
            if e == -1:
                break
            yield buf[s:e + 2]
            buf = buf[e + 2:]


# ================== CAR WRAPPER ==================
class Car:
    def __init__(self):
        self.px = Picarx()

        # camera API from your stare_at_you.py
        if not hasattr(self.px, "set_cam_pan_angle") or not hasattr(self.px, "set_cam_tilt_angle"):
            print("[WARN] Picarx camera pan/tilt functions not found!")

        self.cam_pan = 0.0
        self.cam_tilt = 0.0

        self.center_all()
        self.stop()

    # drive
    def forward(self, pwr):
        self.px.forward(int(clamp(pwr, 0, 100)))

    def backward(self, pwr):
        self.px.backward(int(clamp(pwr, 0, 100)))

    def stop(self):
        self.px.stop()

    def steer(self, deg):
        deg = float(clamp(deg, -35, 35))
        self.px.set_dir_servo_angle(deg)

    def distance_cm(self):
        try:
            d = self.px.ultrasonic.read()
            return float(d) if d and d > 0 else 999.0
        except Exception:
            return 999.0

    # camera servos
    def set_cam_angles(self, pan_deg, tilt_deg):
        self.cam_pan = clamp(pan_deg, -CAM_LIMIT, CAM_LIMIT)
        self.cam_tilt = clamp(tilt_deg, -CAM_LIMIT, CAM_LIMIT)
        try:
            self.px.set_cam_pan_angle(self.cam_pan)
            self.px.set_cam_tilt_angle(self.cam_tilt)
        except Exception as e:
            print("[WARN] set_cam_* failed:", e)

    def nudge_cam_slow(self, cx, cy):
        """
        Low-gain pan/tilt:
        - normalized error [-1,1] in x & y
        - per-frame step is clamped to ±CAM_MAX_STEP deg
        """
        ex = (cx - WIDTH / 2.0) / (WIDTH / 2.0)   # -1 .. 1
        ey = (cy - HEIGHT / 2.0) / (HEIGHT / 2.0)

        pan_step = clamp(ex * CAM_MAX_STEP, -CAM_MAX_STEP, CAM_MAX_STEP)
        tilt_step = clamp(-ey * CAM_MAX_STEP, -CAM_MAX_STEP, CAM_MAX_STEP)

        self.set_cam_angles(self.cam_pan + pan_step,
                            self.cam_tilt + tilt_step)

    def center_cam(self):
        self.set_cam_angles(0.0, 0.0)

    def center_all(self):
        self.center_cam()
        self.steer(0.0)


# ================== FIRE LOCK STATE ==================
class FireLock:
    def __init__(self):
        self.ts = 0.0
        self.cx = self.cy = None
        self.area = 0.0
        self.conf = 0.0

    def update(self, cx, cy, area, conf):
        self.ts = time.time()
        self.cx = cx
        self.cy = cy
        self.area = area
        self.conf = conf

    def mode_target(self):
        age = time.time() - self.ts
        if age <= LOCK_TIMEOUT_S and self.conf >= CONF_THRESH:
            return "TRACK", self.cx, self.cy, self.area
        return "WANDER", None, None, 0.0


# ================== MAIN ==================
def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    model = YOLO(MODEL_PATH).to(device)

    car = Car()
    lock = FireLock()

    # Headless if there is no DISPLAY, or explicitly offscreen
    HEADLESS = not os.environ.get("DISPLAY") or os.environ.get("QT_QPA_PLATFORM") == "offscreen"
    if HEADLESS:
        print("[INFO] Running in HEADLESS mode (no OpenCV window).")

    print("[INFO] starting rpicam-vid (MJPEG, low-latency) ...")
    proc = subprocess.Popen(
        RPICAM_CMD,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0
    )

    if not HEADLESS:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # Wander FSM
    state = "CRUISE"  # CRUISE, CAUTION, DANGER_BACK
    state_until = 0.0
    caution_dir = +1
    prev_mode = "WANDER"

    try:
        for jpeg in mjpeg_frames(proc.stdout):
            frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # -------- YOLO --------
            r = model.predict(
                frame, imgsz=IMGSZ, conf=CONF_THRESH, device=device, verbose=False
            )[0]

            best = None
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    conf = float(b.conf[0].item()) if b.conf is not None else 0.0
                    if conf < CONF_THRESH:
                        continue
                    w = max(1.0, x2 - x1)
                    h = max(1.0, y2 - y1)
                    area = w * h
                    cx = x1 + w / 2.0
                    cy = y1 + h / 2.0
                    if best is None or area > best[-2]:
                        best = (cx, cy, area, conf, (int(x1), int(y1), int(x2), int(y2)))

            if best:
                cx, cy, area, conf, bb = best
                lock.update(cx, cy, area, conf)
                x1, y1, x2, y2 = bb
                if not HEADLESS:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame, f"FIRE {conf:.2f}",
                        (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2, cv2.LINE_AA
                    )

            mode, cx, cy, area = lock.mode_target()

            # recenter gimbal when leaving TRACK
            if prev_mode == "TRACK" and mode == "WANDER":
                car.center_cam()

            if mode == "TRACK" and cx is not None and cy is not None:
                # -------- TRACK: slow pan/tilt follow --------
                car.nudge_cam_slow(cx, cy)

                # Only creep forward if roughly centered (small error)
                err_x = abs(cx - WIDTH / 2.0)
                err_y = abs(cy - HEIGHT / 2.0)
                centered = (err_x < WIDTH * 0.18) and (err_y < HEIGHT * 0.18)

                norm_area = area / float(WIDTH * HEIGHT)
                raw_speed = TRACK_BASE_SPEED + (TRACK_MAX_SPEED - TRACK_BASE_SPEED) * (
                    1.0 - min(1.0, 8.0 * norm_area)
                )
                speed = int(clamp(raw_speed, TRACK_BASE_SPEED, TRACK_MAX_SPEED))

                d = car.distance_cm()
                if centered and d > DANGER_CM:
                    car.steer(0.0)
                    car.forward(speed)
                else:
                    car.stop()

                if not HEADLESS:
                    cv2.putText(
                        frame,
                        f"MODE TRACK | pan={car.cam_pan:.1f} tilt={car.cam_tilt:.1f} spd={speed} d={int(d)}cm",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (40, 220, 40), 2, cv2.LINE_AA
                    )

            else:
                # -------- WANDER: slow safe/caution/danger FSM --------
                d = car.distance_cm()
                now = time.time()

                if state == "CRUISE":
                    if d >= SAFE_CM:
                        car.steer(0.0)
                        car.forward(CRUISE_PWR)
                    elif d >= DANGER_CM:
                        caution_dir = random.choice([+1, -1])
                        car.steer(caution_dir * CAUTION_STEER)
                        car.forward(CAUTION_PWR)
                        state = "CAUTION"
                        state_until = now + CAUTION_HOLD_S
                    else:
                        car.steer(-caution_dir * HARD_TURN)
                        car.backward(BACK_PWR)
                        state = "DANGER_BACK"
                        state_until = now + BACKUP_S

                elif state == "CAUTION":
                    if now >= state_until:
                        state = "CRUISE"
                    else:
                        if d < DANGER_CM:
                            car.steer(-caution_dir * HARD_TURN)
                            car.backward(BACK_PWR)
                            state = "DANGER_BACK"
                            state_until = now + BACKUP_S

                elif state == "DANGER_BACK":
                    if now >= state_until:
                        car.stop()
                        car.steer(-caution_dir * HARD_TURN)
                        car.forward(CAUTION_PWR)
                        time.sleep(0.25)
                        state = "CRUISE"

                if not HEADLESS:
                    cv2.putText(
                        frame,
                        f"MODE WANDER | state={state} d={int(d)}cm",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200, 200, 60), 2, cv2.LINE_AA
                    )

            prev_mode = mode

            # crosshair & display (only if GUI is available)
            if not HEADLESS:
                cv2.drawMarker(
                    frame,
                    (WIDTH // 2, HEIGHT // 2),
                    (255, 255, 255),
                    cv2.MARKER_CROSS,
                    20,
                    2
                )
                cv2.imshow(WINDOW_NAME, frame)
                if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
                    break
            else:
                # tiny sleep to avoid pegging CPU
                time.sleep(0.01)

    finally:
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            car.stop()
            car.center_all()
        except Exception:
            pass
        if not HEADLESS:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# cd /home/pi/Desktop/AFV_FIRE_AI
# source venv/bin/activate
# export QT_QPA_PLATFORM=xcb
# python move_track_picarx.py
