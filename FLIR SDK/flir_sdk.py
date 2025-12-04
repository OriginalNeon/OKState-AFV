#!/usr/bin/env python3
import argparse, sys, time, os, ctypes, threading
import numpy as np
import cv2

# ============================================================
#  CH341 + PCA9685 Servo Turret Controller
# ============================================================

# *** Adjust DLL_PATH if you move the project folder ***
DLL_PATH = r"C:\Users\Enrique Rosa-Berrios\OneDrive - Oklahoma A and M System\Documents\Capstone\afv\CH341DLLA64.DLL"

PCA9685_ADDR = 0x40  # 7-bit I2C address

MODE1     = 0x00
PRESCALE  = 0xFE
LED0_ON_L = 0x06
PRESCALE_50HZ = 121   # 50 Hz for PCA9685 internal 25 MHz clock


class TurretController:
    """
    Slow, smoothed pan/tilt control for the FLIR turret via CH341 + PCA9685.
    State machine:
      WARMUP -> SCAN_DOWN -> SCAN_PAN -> TRACK
    """
    def __init__(self):
        self.ch341 = None
        self.ready = False

        # Servo pulse settings (same as your test script)
        self.center = 307
        self.pan = self.center
        self.tilt = self.center

        # Hard mechanical limits (in PCA9685 counts)
        self.PAN_MIN = 220
        self.PAN_MAX = 400

        # Protect wiring: limit how far "up" we can tilt (≈ < 45° above horizontal)
        # For this board, assume:
        #  - "Down" is numerically smaller than center (tilt--)
        #  - "Up"   is numerically larger than center (tilt++)
        self.TILT_UP_LIMIT   = self.center + 35   # tighten as needed
        self.TILT_DOWN_LIMIT = self.center - 80   # look down toward floor

        # Derived: which numeric direction is "down"?
        diff = self.TILT_DOWN_LIMIT - self.center
        self._tilt_down_sign = float(np.sign(diff) if diff != 0 else -1.0)
        print(f"[INFO] Tilt 'down' numeric sign = {self._tilt_down_sign} "
              f"(center={self.center}, down_limit={self.TILT_DOWN_LIMIT})")

        # Tracking behaviour
        self.dead_px = 15             # pixel dead-zone so it stops once near center
        self.track_k  = 0.0035        # gain: pixels -> PWM counts
        self.track_max_step = 1.5     # max counts per frame (smooth, no jerks)

        # Scan behaviour
        self.warmup_s   = 2.0         # seconds to sit centered at start
        self.no_tgt_timeout_s = 2.0   # time with no target before re-scan
        self.scan_tilt_step = 0.3     # slow tilt while searching
        self.scan_pan_step  = 0.4     # slow left/right sweep
        self.scan_pan_dir   = -1      # start sweeping left

        self.state = "WARMUP"
        self.start_time = None
        self.last_target_time = None

        # CH341 init
        try:
            if not os.path.exists(DLL_PATH):
                print(f"[WARN] CH341 DLL not found at: {DLL_PATH}")
                return

            ch341 = ctypes.WinDLL(DLL_PATH)

            ch341.CH341OpenDevice.argtypes  = [ctypes.c_ulong]
            ch341.CH341OpenDevice.restype   = ctypes.c_long
            ch341.CH341CloseDevice.argtypes = [ctypes.c_ulong]
            ch341.CH341CloseDevice.restype  = None

            # CH341WriteI2C(iIndex, iDevAddr, iRegAddr, iByte)
            ch341.CH341WriteI2C.argtypes = [
                ctypes.c_ulong, ctypes.c_ulong,
                ctypes.c_ulong, ctypes.c_ulong
            ]
            ch341.CH341WriteI2C.restype = ctypes.c_ulong

            handle = ch341.CH341OpenDevice(0)
            if handle < 0:
                print("[WARN] Could not open CH341A device")
                return

            self.ch341 = ch341
            self.ready = True
            print(f"[INFO] CH341 opened for turret control, handle={handle}")

            self._pca9685_init()
            # Center on startup
            self._apply_pwm()
            print("[INFO] Turret centered (startup).")

        except Exception as e:
            print("[WARN] Turret init failed:", e)
            self.ch341 = None
            self.ready = False

    # ---- low-level helpers ----

    def _i2c_write_byte(self, dev_addr, reg, value):
        if not self.ready:
            return
        _ = self.ch341.CH341WriteI2C(0, dev_addr, reg, value & 0xFF)

    def _pca9685_init(self):
        """Configure PCA9685 for 50 Hz servo pulses."""
        if not self.ready:
            return
        print("[INFO] Initializing PCA9685 @ 0x40 for 50 Hz servos...")
        time.sleep(0.01)
        # Reset MODE1
        self._i2c_write_byte(PCA9685_ADDR, MODE1, 0x00)
        time.sleep(0.01)
        # Enter sleep to set prescale
        self._i2c_write_byte(PCA9685_ADDR, MODE1, 0x10)
        time.sleep(0.01)
        # Set prescale
        self._i2c_write_byte(PCA9685_ADDR, PRESCALE, PRESCALE_50HZ)
        time.sleep(0.01)
        # Wake with auto-increment
        self._i2c_write_byte(PCA9685_ADDR, MODE1, 0x20)
        time.sleep(0.05)
        print("[INFO] PCA9685 initialized.")

    def _set_pwm_channel(self, channel, on, off):
        if not self.ready:
            return
        base = LED0_ON_L + 4 * channel
        self._i2c_write_byte(PCA9685_ADDR, base,     on & 0xFF)
        self._i2c_write_byte(PCA9685_ADDR, base + 1, (on >> 8) & 0xFF)
        self._i2c_write_byte(PCA9685_ADDR, base + 2, off & 0xFF)
        self._i2c_write_byte(PCA9685_ADDR, base + 3, (off >> 8) & 0xFF)

    def _apply_pwm(self):
        """Send current pan/tilt pulses to channels 1 (pan) & 0 (tilt)."""
        if not self.ready:
            return
        self._set_pwm_channel(0, 0, int(round(self.tilt)))
        self._set_pwm_channel(1, 0, int(round(self.pan)))

    def close(self):
        if self.ch341 is not None:
            try:
                self.ch341.CH341CloseDevice(0)
                print("[INFO] CH341 closed (turret).")
            except Exception:
                pass
            self.ch341 = None
        self.ready = False

    # ---- high-level behaviour ----

    def _clamp_limits(self):
        # Use numeric min/max so we don't care whether up/down are smaller/bigger
        pan_lo = min(self.PAN_MIN, self.PAN_MAX)
        pan_hi = max(self.PAN_MIN, self.PAN_MAX)
        tilt_lo = min(self.TILT_UP_LIMIT, self.TILT_DOWN_LIMIT)
        tilt_hi = max(self.TILT_UP_LIMIT, self.TILT_DOWN_LIMIT)

        self.pan  = float(np.clip(self.pan,  pan_lo,  pan_hi))
        self.tilt = float(np.clip(self.tilt, tilt_lo, tilt_hi))

    def update(self, control_pt, frame_shape, has_target, now, send_enabled=True):
        """
        Called every frame from the main loop.

        control_pt : (x,y) in image coordinates (base of fire for aim_mode=base).
        has_target : True only if we see a *current* fire/base this frame.
        """
        if not self.ready:
            return

        if self.start_time is None:
            self.start_time = now
            self.last_target_time = now

        if has_target:
            self.last_target_time = now

        # ---------- State transitions ----------
        if self.state == "WARMUP":
            # Hold centered for warmup_s, ALWAYS go to SCAN_DOWN after
            self.pan = self.center
            self.tilt = self.center
            if now - self.start_time >= self.warmup_s:
                self.state = "SCAN_DOWN"

        elif self.state in ("SCAN_DOWN", "SCAN_PAN"):
            if has_target and control_pt is not None:
                self.state = "TRACK"

        elif self.state == "TRACK":
            # If we lose target for a while, restart search
            if (not has_target) and (now - self.last_target_time > self.no_tgt_timeout_s):
                self.state = "SCAN_DOWN"

        # ---------- Behaviour per state ----------
        if self.state == "WARMUP":
            pass

        elif self.state == "SCAN_DOWN":
            # Tilt in the numeric direction we've defined as "down"
            down_sign = self._tilt_down_sign
            self.tilt += down_sign * self.scan_tilt_step

            if down_sign > 0:
                if self.tilt >= self.TILT_DOWN_LIMIT - 0.5:
                    self.tilt = self.TILT_DOWN_LIMIT
                    self.state = "SCAN_PAN"
            else:
                if self.tilt <= self.TILT_DOWN_LIMIT + 0.5:
                    self.tilt = self.TILT_DOWN_LIMIT
                    self.state = "SCAN_PAN"

        elif self.state == "SCAN_PAN":
            # slow left/right sweep around center
            self.pan += self.scan_pan_dir * self.scan_pan_step
            if self.pan <= self.PAN_MIN:
                self.pan = self.PAN_MIN
                self.scan_pan_dir = +1
            elif self.pan >= self.PAN_MAX:
                self.pan = self.PAN_MAX
                self.scan_pan_dir = -1

        elif self.state == "TRACK" and has_target and control_pt is not None:
            h, w = frame_shape[:2]
            cx0, cy0 = w // 2, h // 2
            dx = float(control_pt[0] - cx0)
            dy = float(control_pt[1] - cy0)

            # Pixel dead-zone so we don't dither once we're lined up
            if abs(dx) < self.dead_px:
                dx = 0.0
            if abs(dy) < self.dead_px:
                dy = 0.0

            # +dx = target to the right -> pan right (smaller pulse)
            delta_pan  = -self.track_k * dx

            # +dy = target below center -> we want to tilt "down"
            down_sign = self._tilt_down_sign
            delta_tilt = down_sign * self.track_k * dy

            # Smooth: cap how much we move per frame
            delta_pan  = float(np.clip(delta_pan,  -self.track_max_step, self.track_max_step))
            delta_tilt = float(np.clip(delta_tilt, -self.track_max_step, self.track_max_step))

            self.pan  += delta_pan
            self.tilt += delta_tilt

        # Apply clamps & send to hardware
        self._clamp_limits()
        if send_enabled:
            self._apply_pwm()


# ============================================================
#  Thermal Vectoring Logic (from your SDK)
# ============================================================

def has_gstreamer():
    try:
        info = cv2.getBuildInformation()
        return "GStreamer:                       YES" in info
    except Exception:
        return False

def open_capture(rtsp_url, use_gstreamer=True, latency_ms=100):
    cap = None
    if use_gstreamer and has_gstreamer():
        gst = (
            f"rtspsrc location={rtsp_url} latency={latency_ms} ! "
            "rtph264depay ! h264parse ! openh264dec ! videoconvert ! "
            "appsink drop=true sync=false"
        )
        cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print("[INFO] Using GStreamer + openh264dec")
            return cap
        else:
            print("[WARN] GStreamer failed; falling back to FFmpeg.")

    # FFmpeg backend
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if cap.isOpened():
        print("[INFO] Using FFmpeg backend.")
        try:
            # Smaller buffer -> less lag / fewer bursts
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Ask decoder for ~30 fps (camera still decides real fps)
            cap.set(cv2.CAP_PROP_FPS, 30)
        except Exception:
            pass
        return cap
    return None


# ============================================================
#  Background Frame Grabber for Smoother FPS
# ============================================================

class FrameGrabber(threading.Thread):
    """
    Separate thread that constantly reads frames from the RTSP stream
    and always exposes the *latest* frame. This avoids backlog and
    makes the visible feed & servo updates respond more smoothly.
    """
    def __init__(self, rtsp_url, force_ffmpeg, latency_ms):
        super().__init__(daemon=True)
        self.rtsp_url = rtsp_url
        self.force_ffmpeg = force_ffmpeg
        self.latency_ms = latency_ms

        self.cap = open_capture(
            self.rtsp_url,
            use_gstreamer=(not self.force_ffmpeg),
            latency_ms=self.latency_ms
        )
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        self.last_ok = 0.0

    def run(self):
        fail_count = 0
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                # Try to reconnect
                time.sleep(0.3)
                self.cap = open_capture(
                    self.rtsp_url,
                    use_gstreamer=(not self.force_ffmpeg),
                    latency_ms=self.latency_ms
                )
                fail_count = 0
                continue

            ok, frame = self.cap.read()
            if not ok or frame is None:
                fail_count += 1
                if fail_count > 30:
                    # Drop and reopen after several bad reads
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None
                    fail_count = 0
                time.sleep(0.01)
                continue

            fail_count = 0
            self.last_ok = time.time()
            with self.lock:
                self.frame = frame
            # Tiny sleep to avoid pegging a CPU core
            time.sleep(0.001)

    def get_latest(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.running = False
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.cap = None


# ---- Image processing helpers (unchanged) ----

def morph_clean(mask, k=3):
    if k <= 1:
        return mask
    kernel = np.ones((k, k), np.uint8)
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    return m

def make_mask_from_thresh(gray, thresh, morph_k=3):
    mask = (gray >= thresh).astype(np.uint8) * 255
    mask = morph_clean(mask, morph_k)
    return mask

def centroid_intensity_weighted(gray, mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    vals = gray[ys, xs].astype(np.float64)
    wsum = float(np.sum(vals))
    if wsum == 0.0:
        return None
    cx = int(np.round(np.sum(xs * vals) / wsum))
    cy = int(np.round(np.sum(ys * vals) / wsum))
    return (cx, cy)

def pca_axis_on_points(xs, ys, weights=None):
    if len(xs) == 0:
        return (0.0, 0.0, 0.0, None, None)
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        wsum = float(np.sum(w))
        if wsum <= 0:
            return (0.0, 0.0, 0.0, None, None)
        cx = float(np.sum(xs * w) / wsum)
        cy = float(np.sum(ys * w) / wsum)
        X = np.stack([xs - cx, ys - cy], axis=0)
        Xw = X * w
        C = (Xw @ X.T) / wsum
    else:
        cx = float(np.mean(xs)); cy = float(np.mean(ys))
        X = np.stack([xs - cx, ys - cy], axis=0)
        C = (X @ X.T) / X.shape[1]

    evals, evecs = np.linalg.eigh(C)
    idx = int(np.argmax(evals))
    vec = evecs[:, idx]
    mag = float(evals[idx])
    n = float(np.hypot(vec[0], vec[1]))
    if n < 1e-9:
        return (0.0, 0.0, 0.0, None, None)
    return (float(vec[0]/n), float(vec[1]/n), mag, cx, cy)

def ema_update(prev, new, alpha):
    if prev is None:
        return np.array(new, dtype=float)
    return (1.0 - alpha) * np.array(prev) + alpha * np.array(new)

def largest_blob(mask):
    if mask is None or mask.size == 0:
        return None, None, None, None
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), 8
    )
    if num <= 1:
        return labels, None, None, None
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_rel = 1 + int(np.argmax(areas))
    ys, xs = np.where(labels == best_rel)
    blob_mask = (labels == best_rel).astype(np.uint8) * 255
    return labels, (best_rel, int(areas[best_rel-1])), (xs, ys), blob_mask

def base_of_blob(xs, ys):
    if xs is None or len(xs) == 0:
        return None
    y_max = int(np.max(ys))
    at_base = np.where(ys == y_max)[0]
    xs_base = xs[at_base]
    xb = int(np.median(xs_base))
    return (xb, y_max)

def connected_to_point_mask(low_mask, seed, dilate_iter=1):
    if seed is None:
        return None
    h, w = low_mask.shape[:2]
    sx, sy = int(seed[0]), int(seed[1])
    work = (low_mask > 0).astype(np.uint8) * 255
    if dilate_iter > 0:
        work = cv2.dilate(work, np.ones((3,3), np.uint8), iterations=dilate_iter)

    if sx < 0 or sy < 0 or sx >= w or sy >= h:
        return None
    if work[sy, sx] == 0:
        y0, y1 = max(0, sy-2), min(h, sy+3)
        x0, x1 = max(0, sx-2), min(w, sx+3)
        sub = np.argwhere(work[y0:y1, x0:x1] > 0)
        if sub.size == 0:
            return None
        dy, dx = sub[0]
        sx = x0 + int(dx); sy = y0 + int(dy)

    ff_mask = np.zeros((h+2, w+2), np.uint8)
    flood = work.copy()
    cv2.floodFill(flood, ff_mask, (sx, sy), 128)
    return (flood == 128).astype(np.uint8) * 255

def plume_axis_and_end(gray, conn_mask, base_pt, up_only=True):
    if conn_mask is None or base_pt is None:
        return None, None, 0.0
    ys, xs = np.where(conn_mask > 0)
    if len(xs) == 0:
        return None, None, 0.0

    weights = gray[ys, xs].astype(np.float64)
    ux, uy, strength, _, _ = pca_axis_on_points(xs, ys, weights=weights)
    if strength <= 0:
        return None, None, 0.0

    if up_only and uy > 0:
        ux, uy = -ux, -uy

    bx, by = float(base_pt[0]), float(base_pt[1])
    dx = xs.astype(np.float64) - bx
    dy = ys.astype(np.float64) - by
    t = dx * ux + dy * uy
    t_pos = t[t > 0]
    if t_pos.size == 0:
        return (ux, uy), (int(bx + 80*ux), int(by + 80*uy)), strength
    t_max = float(np.max(t_pos))
    end = (int(round(bx + t_max*ux)), int(round(by + t_max*uy)))
    return (ux, uy), end, strength

def draw_vectors(frame_bgr,
                 aim_point, base_pt, plume_end,
                 line_thick=2, outline_thick=4,
                 tip_green=0.08, tip_blue=0.08):
    h, w = frame_bgr.shape[:2]
    cx0, cy0 = w // 2, h // 2

    cv2.drawMarker(frame_bgr, (cx0, cy0), (255,255,255),
                   cv2.MARKER_CROSS, 18, 1)

    if aim_point is not None:
        c = (int(round(aim_point[0])), int(round(aim_point[1])))
        cv2.circle(frame_bgr, c, 3, (0,0,255), -1)
        cv2.arrowedLine(frame_bgr, (cx0, cy0), c,
                        (0,255,0), line_thick, tipLength=tip_green)

    if base_pt is not None and plume_end is not None:
        cv2.arrowedLine(frame_bgr, base_pt, plume_end,
                        (255,255,255), outline_thick, tipLength=tip_blue)
        cv2.arrowedLine(frame_bgr, base_pt, plume_end,
                        (255,255,0),   line_thick,   tipLength=tip_blue)
    return frame_bgr

def draw_hud_top_left_on_canvas(canvas_bgr, fps, thresh, centroid_for_hud,
                                pan_cmd, tilt_cmd, pca_strength, plume_angle_deg):
    lines = [
        f"FPS: {fps:5.1f}",
        f"Thresh: {thresh} (+/-)",
        f"Centroid: {None if centroid_for_hud is None else (int(round(centroid_for_hud[0])), int(round(centroid_for_hud[1])))}",
        f"Cmd: pan={pan_cmd:+.3f}, tilt={tilt_cmd:+.3f}",
        f"PCA strength: {pca_strength:.2f}" if pca_strength is not None else "PCA strength: n/a",
        f"Plume angle: {plume_angle_deg:5.1f} deg" if plume_angle_deg is not None else "Plume angle: n/a",
    ]
    y = 20
    for line in lines:
        cv2.putText(canvas_bgr, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255,255,255), 1, cv2.LINE_AA)
        y += 22

def draw_keys_bottom_left(canvas_bgr):
    keys_line = "Keys: q=quit  +/-=thresh  r=reset  g=toggle send  f=fullscreen"
    y = int(canvas_bgr.shape[0] - 12)
    cv2.putText(canvas_bgr, keys_line, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255,255,255), 1, cv2.LINE_AA)

def compute_commands(target_pt, shape, kx, ky, dead_px):
    if target_pt is None:
        return 0.0, 0.0
    h, w = shape[:2]
    cx0, cy0 = w // 2, h // 2
    dx = target_pt[0] - cx0
    dy = target_pt[1] - cy0
    if abs(dx) < dead_px: dx = 0
    if abs(dy) < dead_px: dy = 0
    pan  = kx * dx
    tilt = ky * dy
    return pan, tilt

# ============================================================
#  Main application (thermal + servo)
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="FLIR A50 thermal vectoring + turret tracking"
    )
    ap.add_argument("--rtsp", required=True)
    ap.add_argument("--thresh", type=int, default=180)
    ap.add_argument("--kx", type=float, default=0.0020)
    ap.add_argument("--ky", type=float, default=0.0020)
    ap.add_argument("--dead", type=int, default=8)
    ap.add_argument("--latency", type=int, default=100)
    ap.add_argument("--force_ffmpeg", action="store_true")

    ap.add_argument("--ema", type=float, default=0.30)
    ap.add_argument("--morphk", type=int, default=5)
    ap.add_argument("--plume_low_offset", type=int, default=45)
    ap.add_argument("--plume_up_only", action="store_true",
                    default=True, dest="plume_up_only")
    ap.add_argument("--no-plume_up_only", action="store_false",
                    dest="plume_up_only")
    ap.add_argument("--aim_mode", choices=["centroid","base"], default="base")

    ap.add_argument("--canvas_w", type=int, default=1280)
    ap.add_argument("--canvas_h", type=int, default=720)

    ap.add_argument("--tip_green", type=float, default=0.08)
    ap.add_argument("--tip_blue",  type=float, default=0.08)
    ap.add_argument("--line_thick", type=int, default=2)
    ap.add_argument("--outline_thick", type=int, default=4)

    ap.add_argument("--axis_alpha", type=float, default=0.35)
    ap.add_argument("--len_alpha",  type=float, default=0.30)
    ap.add_argument("--len_rate_limit", type=float, default=0.25)

    ap.add_argument("--display_interp", choices=["linear", "nearest"],
                    default="linear")
    ap.add_argument("--rect_poll_hz", type=float, default=4.0)
    ap.add_argument("--conn_dilate_iter", type=int, default=1)

    ap.add_argument("--proc_scale", type=float, default=0.5)
    ap.add_argument("--plume_every", type=int, default=2)

    args = ap.parse_args()

    args.proc_scale = float(np.clip(args.proc_scale, 0.25, 1.0))
    args.plume_every = max(1, int(args.plume_every))

    # Turret controller
    turret = TurretController()

    # Start frame grabber thread
    grabber = FrameGrabber(args.rtsp, args.force_ffmpeg, args.latency)
    grabber.start()
    time.sleep(0.5)  # let it warm up a bit

    print("[INFO] 'f' = fullscreen, 'g' = toggle servo send, 'q' = quit.")

    t0 = time.time()
    frames = 0
    send_enabled = True
    centroid_ema = None

    u_prev = None
    t_prev = None
    plume_angle_deg = None
    last_plume_end_full = None
    last_base_full = None

    window_name = "FLIR A50 SDK (Thermal Vectoring + Turret Tracking)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.canvas_w, args.canvas_h)
    is_fullscreen = False

    win_rect_cache = (args.canvas_w, args.canvas_h)
    last_rect_poll = 0.0
    rect_poll_interval = 1.0 / max(0.1, args.rect_poll_hz)
    interp_flag = cv2.INTER_LINEAR if args.display_interp == "linear" else cv2.INTER_NEAREST

    while True:
        frame = grabber.get_latest()
        if frame is None:
            # No frame yet, or reconnecting – just wait a moment
            time.sleep(0.01)
            continue

        frames += 1
        now = time.time()

        h_full, w_full = frame.shape[:2]
        if args.proc_scale < 1.0:
            gray_proc = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                (int(w_full*args.proc_scale), int(h_full*args.proc_scale)),
                interpolation=cv2.INTER_AREA
            )
        else:
            gray_proc = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        hot_mask = make_mask_from_thresh(gray_proc, args.thresh, morph_k=args.morphk)
        _, best_blob_info, xy, _ = largest_blob(hot_mask)

        # ---- Base of current blob (in proc space) ----
        base_pt_proc = None
        if best_blob_info is not None and xy is not None:
            xs_blob, ys_blob = xy
            base_pt_proc = base_of_blob(xs_blob, ys_blob)

        # --- “Current” base for control vs “remembered” base for drawing ---
        base_pt_full_current = None   # used for servo tracking
        target_visible = False        # True only if we saw a base this frame

        if base_pt_proc is not None:
            target_visible = True
            if args.proc_scale < 1.0:
                base_pt_full_current = (
                    int(round(base_pt_proc[0] / args.proc_scale)),
                    int(round(base_pt_proc[1] / args.proc_scale))
                )
            else:
                base_pt_full_current = base_pt_proc
            last_base_full = base_pt_full_current  # remember for drawing

        # base point to *draw* can be the last known base
        base_pt_full_draw = last_base_full

        centroid_raw_proc = centroid_intensity_weighted(gray_proc, hot_mask)
        centroid_use_full = None
        if centroid_raw_proc is not None:
            if args.proc_scale < 1.0:
                centroid_raw_full = (centroid_raw_proc[0] / args.proc_scale,
                                     centroid_raw_proc[1] / args.proc_scale)
            else:
                centroid_raw_full = centroid_raw_proc
            centroid_ema = ema_update(centroid_ema, centroid_raw_full, args.ema)
            centroid_use_full = (float(centroid_ema[0]), float(centroid_ema[1]))

        plume_end_full = None
        pca_strength = 0.0

        if (frames % args.plume_every == 0) and (base_pt_proc is not None):
            low_thr = max(0, args.thresh - args.plume_low_offset)
            low_mask = make_mask_from_thresh(
                gray_proc, low_thr, morph_k=max(1, args.morphk//2)
            )
            conn_mask = connected_to_point_mask(
                low_mask, base_pt_proc,
                dilate_iter=max(0, args.conn_dilate_iter)
            )

            uvec, end_pt_proc, pca_strength = plume_axis_and_end(
                gray_proc, conn_mask, base_pt_proc,
                up_only=args.plume_up_only
            )

            if uvec is not None and end_pt_proc is not None:
                if args.proc_scale < 1.0:
                    bx_full = float(base_pt_proc[0] / args.proc_scale)
                    by_full = float(base_pt_proc[1] / args.proc_scale)
                    ex_full = float(end_pt_proc[0] / args.proc_scale)
                    ey_full = float(end_pt_proc[1] / args.proc_scale)
                else:
                    bx_full, by_full = float(base_pt_proc[0]), float(base_pt_proc[1])
                    ex_full, ey_full = float(end_pt_proc[0]), float(end_pt_proc[1])

                ux, uy = uvec
                u_curr = np.array([ux, uy], dtype=float)
                if u_prev is not None and float(np.dot(u_curr, u_prev)) < 0.0:
                    u_curr = -u_curr
                if u_prev is None:
                    u_prev = u_curr
                else:
                    u_prev = (1.0 - args.axis_alpha) * u_prev + args.axis_alpha * u_curr
                    n = float(np.hypot(u_prev[0], u_prev[1]))
                    if n > 1e-6:
                        u_prev /= n

                t_raw = (ex_full - bx_full) * u_prev[0] + (ey_full - by_full) * u_prev[1]
                if t_prev is None:
                    t_prev = t_raw
                else:
                    t_target = (1.0 - args.len_alpha) * t_prev + args.len_alpha * t_raw
                    max_step = max(5.0, abs(t_prev) * args.len_rate_limit)
                    t_prev = t_prev + np.clip(t_target - t_prev, -max_step, max_step)

                exs = int(round(bx_full + t_prev * u_prev[0]))
                eys = int(round(by_full + t_prev * u_prev[1]))
                plume_end_full = (exs, eys)
                last_plume_end_full = plume_end_full

                vx = exs - bx_full
                vy = eys - by_full
                if not (vx == 0.0 and vy == 0.0):
                    ang = float(np.degrees(np.arctan2(-vy, vx)))
                    if ang < 0.0:
                        ang += 360.0
                    if ang > 180.0:
                        ang = 360.0 - ang
                    plume_angle_deg = ang
        else:
            plume_end_full = last_plume_end_full

        # ---- Choose control point for servo (green arrow base) ----
        if args.aim_mode == "base" and base_pt_full_current is not None:
            control_pt = base_pt_full_current
        elif args.aim_mode == "centroid" and centroid_use_full is not None:
            control_pt = centroid_use_full
        else:
            control_pt = None

        pan_cmd, tilt_cmd = compute_commands(
            control_pt, frame.shape, args.kx, args.ky, args.dead
        )
        fps = frames / max(now - t0, 1e-6)

        # ---- Turret update: smooth follow of green arrow ----
        has_target_raw = target_visible and (control_pt is not None)
        has_target = has_target_raw

        if turret is not None and turret.ready:
            # Only use "floor gating" during search states.
            if turret.state in ("WARMUP", "SCAN_DOWN", "SCAN_PAN"):
                if has_target and args.aim_mode == "base" and base_pt_full_current is not None:
                    h_frame = frame.shape[0]
                    floor_min_y = int(0.60 * h_frame)  # bottom ~40% of image
                    if control_pt[1] < floor_min_y:
                        has_target = False  # ignore hot stuff not near the floor

            # Once in TRACK, we pass has_target_raw (no gating) to stay locked.
            if turret.state == "TRACK":
                has_target = has_target_raw

            turret.update(control_pt, frame.shape, has_target,
                          now, send_enabled=send_enabled)

        disp = frame.copy()
        disp = draw_vectors(
            disp, control_pt, base_pt_full_draw, plume_end_full,
            line_thick=args.line_thick,
            outline_thick=args.outline_thick,
            tip_green=args.tip_green,
            tip_blue=args.tip_blue
        )

        now_t = time.time()
        if now_t - last_rect_poll >= rect_poll_interval:
            try:
                _, _, win_w, win_h = cv2.getWindowImageRect(window_name)
                if win_w > 0 and win_h > 0:
                    win_rect_cache = (win_w, win_h)
            except Exception:
                pass
            last_rect_poll = now_t
        canvas_w, canvas_h = win_rect_cache

        h, w = disp.shape[:2]
        scale = min(canvas_w / w, canvas_h / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        disp_scaled = cv2.resize(disp, (new_w, new_h),
                                 interpolation=interp_flag)

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        offset_x = (canvas_w - new_w) // 2
        offset_y = (canvas_h - new_h) // 2
        canvas[offset_y:offset_y+new_h,
               offset_x:offset_x+new_w] = disp_scaled

        draw_hud_top_left_on_canvas(
            canvas, fps, args.thresh, centroid_use_full,
            pan_cmd, tilt_cmd, pca_strength, plume_angle_deg
        )
        draw_keys_bottom_left(canvas)

        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('+') or key == ord('='):
            args.thresh = min(args.thresh + 5, 255)
        elif key == ord('-') or key == ord('_'):
            args.thresh = max(args.thresh - 5, 0)
        elif key == ord('r'):
            args.thresh = 180
        elif key == ord('g'):
            send_enabled = not send_enabled
            print(f"[INFO] Send servo: {send_enabled}")
        elif key == ord('f'):
            is_fullscreen = not is_fullscreen
            cv2.setWindowProperty(
                window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
            )
            if not is_fullscreen:
                cv2.resizeWindow(window_name, args.canvas_w, args.canvas_h)

    grabber.stop()
    if turret is not None:
        turret.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
