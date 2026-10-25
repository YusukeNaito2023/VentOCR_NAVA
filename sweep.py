"""Sweep (pen-tip) detection and tracking for the Servo-U waveform screen.

The Servo-U draws the four waveforms from left to right at a constant speed. Just ahead of the pen an erase band
(about 38 px at 1280x720) removes the previous sweep. When the pen reaches the right edge it jumps back to the left
edge and the whole waveform area is redrawn (the y-axis scales may change at that moment).

- SweepDetector runs inside initial_ocr on every frame (the frames are in memory). It compares a few rows of the
  Pressure / Flow / Volume bands with the previous frame and records the pen tip, in pixels of the frame it received.
  A normal frame changes in two runs of columns: the newly drawn trace (its right edge is the pen tip) and, about one
  erase-band width to the right, the erased old trace. Frames where the whole area changes (redraw at a wrap,
  pop-ups) give no value.
- fit_sweep runs in post_ocr. The sweep speed is constant, so tip ~ speed * i + phase (mod W). It joins robust
  per-block phases, corrects the speed from their slope and returns, for every frame, the pen tip as an unwrapped
  position (number of columns drawn so far) in 1280x720 coordinates.
"""
import cv2
import numpy as np

from . import Settings

# Geometry at Settings.BASE_RESOLUTION (1280x720). Measured on Servo-U recordings (1280x720 and 1920x1080).
SWEEP_X_LEFT = 140          # left edge of the drawing area (first column after a wrap)
SWEEP_X_RIGHT = 889         # SWEEP_X_LEFT + one sweep width (749 px); the pen jumps back to the left edge here
SWEEP_W = SWEEP_X_RIGHT - SWEEP_X_LEFT
SWEEP_PX_PER_SEC = 81.6     # sweep speed used when it cannot be estimated (81.5-81.7 on 3 recordings)
ERASE_PX = 38               # erase band: distance from the pen tip to the right edge of the erased columns
DETECT_BANDS = ((89, 219), (225, 355), (362, 492))   # Pressure / Flow / Volume bands without their border lines
DETECT_ROW_STEP = 2         # use every 2nd row of the bands (base px)
DIFF_THRESH = 25            # gray-level change that counts as a changed pixel
SMOOTH_PX = 5               # column smoothing of the per-band change (base px)
RUN_GAP_PX = 6              # changed columns closer than this form one run (base px)
MAX_RUN_PX = 40             # a run wider than this is not a pen/erase run (redraw, pop-up)
PAIR_MIN, PAIR_MAX = 25, 55 # allowed distance between the right edges of the trace run and the erase run (base px)


class SweepDetector:
    """Per-frame pen-tip detector (initial_ocr). add(frame) returns the pen tip x in pixels of the given frame, or NaN."""

    def __init__(self):
        self.prev = None
        self.geom = None

    def _setup(self, shape):
        h, w = shape[:2]
        bw, bh = Settings.BASE_RESOLUTION
        sx, sy = w / bw, h / bh
        rows = sorted({int(round(y * sy)) for y0, y1 in DETECT_BANDS for y in range(y0, y1, DETECT_ROW_STEP)})
        band_of_row = np.array([next(k for k, (y0, y1) in enumerate(DETECT_BANDS) if y0 * sy - 1 <= r <= y1 * sy + 1)
                                for r in rows])
        x0, x1 = int(round((SWEEP_X_LEFT - 10) * sx)), int(round((SWEEP_X_RIGHT + 12) * sx))
        k = max(1, int(round(SMOOTH_PX * sx)))
        self.geom = dict(sx=sx, rows=np.array(rows), band_of_row=band_of_row, x0=x0, x1=min(x1, w), kernel=np.ones(k) / k,
                         gap=RUN_GAP_PX * sx, max_run=MAX_RUN_PX * sx, pair=(PAIR_MIN * sx, PAIR_MAX * sx),
                         right_zone=(SWEEP_X_RIGHT - 50) * sx)

    def add(self, frame):
        if self.geom is None or self.geom.get("shape") != frame.shape[:2]:
            self._setup(frame.shape)
            self.geom["shape"] = frame.shape[:2]
            self.prev = None
        g = self.geom
        strip = frame[g["rows"], g["x0"]:g["x1"]]
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY).astype(np.int16)
        prev, self.prev = self.prev, gray
        if prev is None or prev.shape != gray.shape:
            return np.nan
        diff = np.abs(gray - prev)
        votes = np.zeros(diff.shape[1], np.int16)
        for k in range(len(DETECT_BANDS)):
            col = diff[g["band_of_row"] == k].max(axis=0)
            votes += np.convolve(col, g["kernel"], mode="same") > DIFF_THRESH
        xs = np.flatnonzero(votes >= 2)
        if len(xs) == 0:
            return np.nan
        runs = []
        start = last = xs[0]
        for x in xs[1:]:
            if x > last + g["gap"]:
                runs.append((start, last))
                start = x
            last = x
        runs.append((start, last))
        if any(b - a > g["max_run"] for a, b in runs) or len(runs) > 3:
            return np.nan                                   # redraw at a wrap, pop-up or scale change
        rights = [g["x0"] + b for a, b in runs]
        lo, hi = g["pair"]
        for r0, r1 in zip(rights, rights[1:]):
            if lo <= r1 - r0 <= hi:
                return float(r0)                            # trace run followed by the erase run
        if len(rights) == 1 and rights[0] >= g["right_zone"]:
            return float(rights[0])                         # near the right edge the erase band is already past it
        return np.nan


def _circ_center(values, period, trim):
    """Outlier-robust circular mean of values with period `period`."""
    ang = values / period * 2 * np.pi
    center = np.angle(np.mean(np.exp(1j * ang)))
    for _ in range(3):
        d = np.angle(np.exp(1j * (ang - center))) / (2 * np.pi) * period
        keep = np.abs(d) <= trim
        if keep.sum() == 0:
            break
        center = np.angle(np.mean(np.exp(1j * ang[keep])))
    return center / (2 * np.pi) * period


def _phase_track(tip, valid, speed, W, window, block, trim):
    """Per-block phase (robust centre of tip - speed*i mod W), joined with the wrap-arounds unwrapped."""
    n = len(tip)
    idx = np.arange(n)
    phase = np.mod(tip - speed * idx, W)
    centers = np.arange(0, n, block)
    ph = np.array([_circ_center(phase[sel], W, trim) if (sel := valid & (np.abs(idx - c) <= window)).sum() >= 5 else np.nan
                   for c in centers], float)
    ok = np.isfinite(ph)
    if not ok.any():
        return None, None
    return centers[ok], np.unwrap(ph[ok] / W * 2 * np.pi) / (2 * np.pi) * W


def fit_sweep(tip_base, fps, trim=25, tol=6):
    """Pen tip of every frame from the per-frame detections (1280x720 x, NaN where not detected).

    Returns dict(u=unwrapped tip position in columns from SWEEP_X_LEFT (never decreasing),
                 x=tip x on the screen (1280x720), speed=px per frame, detected=frames whose own detection was used)."""
    W = SWEEP_W
    window = max(1, round(15.0 * fps))
    block = max(1, round(2.5 * fps))
    n = len(tip_base)
    tip = np.asarray(tip_base, float) - SWEEP_X_LEFT
    valid = np.isfinite(tip)
    idx = np.arange(n)
    speed = SWEEP_PX_PER_SEC / fps
    if valid.sum() < 10:
        u = speed * idx
        return {"u": u, "x": SWEEP_X_LEFT + np.mod(u, W), "speed": speed, "detected": np.zeros(n, bool)}

    lag = max(1, round(2.0 * fps))
    if n > lag:
        both = valid[:-lag] & valid[lag:]
        if both.sum() >= 20:
            est = np.median(np.mod(tip[lag:][both] - tip[:-lag][both], W)) / lag
            if 0 < est * lag < W / 2:
                speed = float(est)
    for _ in range(3):
        centers, ph_u = _phase_track(tip, valid, speed, W, window, block, trim)
        if centers is None or len(centers) < 3:
            break
        keep = np.ones(len(centers), bool)
        for _ in range(3):
            slope, icpt = np.polyfit(centers[keep], ph_u[keep], 1)
            r = ph_u - (slope * centers + icpt)
            keep = np.abs(r) <= max(3 * np.std(r[keep]), 5)
        speed += float(slope)
        if abs(slope) < 1e-4:
            break
    centers, ph_u = _phase_track(tip, valid, speed, W, window, block, trim)
    model = speed * idx + (np.interp(idx, centers, ph_u) if centers is not None else 0.0)
    model = model - np.floor(model[0] / W) * W
    u = model.copy()
    d = np.mod(tip - model + W / 2, W) - W / 2
    use = valid & (np.abs(d) <= tol)
    u[use] = model[use] + d[use]
    u = np.maximum.accumulate(u)
    return {"u": u, "x": SWEEP_X_LEFT + np.mod(u, W), "speed": speed, "detected": use}
