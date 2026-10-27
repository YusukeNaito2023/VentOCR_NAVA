import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import Global_Setting as GS
from . import Settings
from . import sweep

# Progress (overall) reporting is best-effort. Main processing continues even if the import fails.
try:
    import progress_io
except Exception:
    progress_io = None


def _report_progress(user_id, analyze_id, *, message, current, total):
    """Forward progress of post_ocr sub-steps to overall (failures are ignored).
    The completion point (current>=total) is always written so throttling cannot swallow it."""
    if progress_io is None or analyze_id is None:
        return
    try:
        progress_io.report_overall(
            user_id, analyze_id, stage="post_processing", status="running",
            message=message, current=current, total=total,
            force=bool(total) and current >= total,
        )
    except Exception:
        pass


def _load_frame(path):
    """Load a frame image and normalize it to the base resolution (Settings.BASE_RESOLUTION).
    For resolution independence, all frames are brought to the base resolution in which ROI/Y_RANGES are defined.
    v1.4: the app generates frame images on the fly (manifest frame_images: on_demand), so always read by passing a path to cv2.imread.
    During BreathsToCSVs, the last few images are cached so the same image is not re-read (the end image is read 3 times)."""
    if _frame_cache is not None and path in _frame_cache:
        _frame_cache.move_to_end(path)
        return _frame_cache[path].copy()
    img = cv2.imread(path)
    if img is None:
        return None
    if os.path.basename(os.path.dirname(os.path.dirname(path))) == KEEP_DIR_NAME:
        canvas = np.zeros((Settings.BASE_RESOLUTION[1], Settings.BASE_RESOLUTION[0], 3), np.uint8)   # kept waveform area
        x1, y1, x2, y2 = KEEP_BOX
        canvas[y1:y2, x1:x2] = img
        img = canvas
    if (img.shape[1], img.shape[0]) != Settings.BASE_RESOLUTION:
        interp = cv2.INTER_AREA if img.shape[1] > Settings.BASE_RESOLUTION[0] else cv2.INTER_LINEAR
        img = cv2.resize(img, Settings.BASE_RESOLUTION, interpolation=interp)
    if _frame_cache is not None:
        _frame_cache[path] = img.copy()
        while len(_frame_cache) > FRAME_CACHE_SIZE:
            _frame_cache.popitem(last=False)
    return img


# 0.2.0: initial_ocr keeps the images post_ocr reads (core._keep_frame), so the app does not decode the video again:
# the waveform area KEEP_BOX (x1, y1, x2, y2 at BASE_RESOLUTION) of the last frame before each display update of
# KEEP_TRIGGER_KEYS, of the frames KEEP_AFTER frames after it, and of the last frame of each segment, in
# <work>/KEEP_DIR_NAME/<segment>/frame_<iFrame>.jpg. Other images (e.g. after a pop-up) come from the app as before.
# 0.2.1: the update frame itself and the 2 frames after it are kept too (in 0.2.0 most images the app still made were
# these: the display update was found 1-2 frames off the frame of the breath boundary)
KEEP_DIR_NAME = "NAVA_Kept"
KEEP_BOX = (130, 80, 900, 640)
KEEP_TRIGGER_KEYS = ("Cdyn", "VTe")
KEEP_AFTER = (0, 1, 2, 5, 10)
START_BACK_SEC = 0.6   # start image is this many seconds before the breath boundary (fixed at 6 frames up to v1.3)
FRAME_CACHE_SIZE = 8   # per breath we read 4 images: start, end, and the one just before each
_frame_cache = None    # OrderedDict only during BreathsToCSVs (path -> base-resolution image)

# --- Whether to perform missing-value interpolation and smoothing ---

TRIM_TO_BREATH = True  # v1.3: trim waveforms to the breath span (inspiration start to next inspiration start)
SMOOTHING = False   # v1.3: smoothing now actually takes effect, so default is OFF (also OFF in engine_options.json)
IMPUTATION = True
# v1.4: abnormal_workers (parallel loading in DetectAbnormalFrames) removed. Ignored if still present in old settings

def set_config(settings):
    global SMOOTHING, IMPUTATION, TRIM_TO_BREATH
    SMOOTHING = settings.get("post_ocr", {}).get("smoothing", False)
    IMPUTATION = settings.get("post_ocr", {}).get("imputation", True)
    TRIM_TO_BREATH = settings.get("post_ocr", {}).get("trim_to_breath", True)

# --- Color and position definitions ---
COLOR_DEFINITIONS = {
    "Pressure": np.array([110, 163, 193]),
    "Flow": np.array([90, 123, 86]),
    "Volume": np.array([114, 118, 89]),
    "Edi": np.array([171, 105, 166])
}

# Calibration bands (vertical) of the waveform panels. Re-measured in Photoshop (on BASE_RESOLUTION=1280x720).
#     Each (top_y, bot_y) corresponds to panel top line = scale max, bottom line = scale min,
#     and is used directly as the calibration anchor of _pixels_to_physical (top=max, bot=min).
#         Pressure: top=+40 line, bot=-4 line (verified consistent with interpolated 0 line y≈208)
#         Flow:     top=+max line, bot=-max line (symmetric axis; 0 line y≈291 matches the midpoint)
#         Volume:   top=scale max line, bot=0 mL line
#         Edi:      top=scale max line, bot=0 µV reference line
#     * Old values had drifted from piecemeal fixes (e.g. resolution changes), so all waveforms were recalibrated by measurement.
#     * get_difference_regions also references this definition as the single source of truth (no duplicate definitions).
Y_RANGES = {
    "Pressure": (86, 222),
    "Flow":     (222, 358),
    "Volume":   (359, 495),
    "Edi":      (496, 631),
}

COLOR_TOLERANCE = 40

# --- Parameters for HSV (hue) based matching ---
# v1.2 changes:
#     v1.1 classified waveform color by BGR Euclidean distance (tolerance COLOR_TOLERANCE), but
#     when downscaling high-res video (1920x1080) to 1280x720, steep rises (near-vertical thin lines)
#     were darkened/blended by anti-aliasing, BGR distance exceeded tolerance and became NaN (rise detection failures).
#     Hue is robust to drops in brightness/saturation and stays stable across devices and downscale blur,
#     so Hue becomes the primary criterion. Saturation/value are used only to exclude background and text.

# Target Hue per waveform (OpenCV scale 0-180). Derived automatically from COLOR_DEFINITIONS, sharing color definitions with v1.1.
def _bgr_to_hue(bgr):
    px = np.uint8([[list(np.asarray(bgr, dtype=np.uint8))]])
    return int(cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0, 0])

TARGET_HUES = {name: _bgr_to_hue(bgr) for name, bgr in COLOR_DEFINITIONS.items()}

HUE_TOLERANCE = 12      # hue tolerance (circular distance, 0-180)
SAT_MIN = 40            # saturation below this is treated as background/gray text and excluded
VAL_MIN = 60            # value below this is treated as background and excluded
VAL_MIN_LEFT = 45       # left edge (140-175) is the label area where waveforms are dark, so relax the value condition
# Fallback for white (tip of the sweep cursor)
WHITE_SAT_MAX = 45
WHITE_VAL_MIN = 140
BORDER_ROWS = 2         # rows excluded from the top of the band (panel border line)
TITLE_ROWS = 14         # rows at the top of the band with the panel title text (not used by the white fallback)

# --- Parameters for determining waveform points from per-column ranges (v1.3) ---
# v1.3 changes:
#     v1.2 took the 'single pixel with the closest hue' per column. At steep parts (e.g. inspiration-to-expiration switch)
#     a column contains a 30-60px vertical line (spread over several columns by downscale/JPEG blur), and which height was chosen
#     was nearly random, so values jumped up/down vs. neighbor columns, producing spikes not on screen (e.g. sample B10 at t≈0.73s).
#     v1.3 takes the 'range the line passed through' per column, and
#         1) among vertically contiguous pixel runs, optimally selects over the whole breath the line connected to neighbor columns (removes noise/text)
#         2) optimally draws a smooth path within the ranges over the whole breath (vertical lines become slopes, no jumps)
#         3) where a range locally protrudes beyond the path (tips of V-shaped troughs/peaks), pins to that value and redraws
#     thereby keeping tips (peak expiratory flow, pressure overshoot) without producing spikes.
RUN_JOIN = 4            # in the same column, gaps smaller than this are joined into one run
SEG_GAP = 12            # if the column-index gap exceeds this, treat as a different line and break the path
RUN_HUE_WEIGHT = 0.5    # weight on hue deviation when selecting runs (connectivity is primary)
PATH_STEP = 0.5         # height step of the path (px)
TIP_MARGIN = 2.0        # columns where the range protrudes beyond the path by at least this are tip candidates (px)
TIP_DROP = 2.0          # on both sides of a tip, the range edge must recede by at least this (i.e. not mid-slope)
TIP_WIN = 8             # number of columns examined on each side of a tip
TIP_SUPPORT = 2         # at least this many columns must reach the tip depth (±TIP_SUPPORT_PX) (excludes single-column text/noise)
TIP_SUPPORT_PX = 1.5


# ============================================================================
#  Axis scale (vertical tick) estimation
#  The axis numbers at the panel's top-left are only 7-8px tall at 1280x720, so OCR almost never works
#  (80->30, 100/110->unreadable). Instead, back-calculate per breath from the large on-screen values on the right and the waveform pixels.
#      Pressure: Real_Ppeak(cmH2O) <-> waveform peak pixel. Proportional by height from the 0 cmH2O baseline (faint horizontal line)
#      Volume:   Real_VTi(ml) <-> waveform peak pixel (0 ml = band bottom)
#      Flow:     no displayed value, so solve so that the integral of Flow pixels matches the Volume (ml) change
#  The resulting 'band top value' is rounded to the tick step (SCALE_SNAP), and outliers removed by the mode of neighboring breaths.
#  With Auto Scaling the ticks change per breath (in the 2026-06-01 video, Flow moved between 60-110 and Volume 550-850),
#  so estimate per breath. Computation is just pixel arithmetic, so doing it for all breaths is cheap.
#  Edi is calibrated by EdiPeak as before (_edi_scale_from_peak).
# ============================================================================
SCALE_UNITS = {"Pressure": "cmH2O", "Flow": "L/min", "Volume": "mL", "Edi": "uV"}
# Step size for manual chart scale adjustment (+/-) (engine-specific).
SCALE_STEPS = {"Pressure": 5, "Flow": 10, "Volume": 100, "Edi": 5}


def scale_meta():
    """Axis-scale metadata for waveform charts (engine-specific). post_ocr saves it to the Archive as scale_meta.json
    and Label (Flask/JS) reads it to apply step size, units and symmetry.
    Keys are lowercase to match wave_data."""
    return {
        "steps":     {k.lower(): v for k, v in SCALE_STEPS.items()},
        "units":     {k.lower(): v for k, v in SCALE_UNITS.items()},
        # Whether the axis is symmetric (min≈-max). True only for Flow (+/- moves both ends symmetrically).
        "symmetric": {k.lower(): k == "Flow" for k in SCALE_UNITS},
    }


# y of the 0 baseline (faint gray horizontal line) (BASE_RESOLUTION). Does not move when ticks change 25/30/40 (verified on 3 videos).
# Per breath, search ±ZERO_SEARCH px in the end image; use this value if not found.
ZERO_Y = {"Pressure": 208, "Flow": 291}
ZERO_SEARCH = 4
# Tick step of the band top value (scale max). Values seen on real screens: Pressure 25/30/40, Flow 50-110, Volume 400-850
SCALE_SNAP = {"Pressure": 5, "Flow": 10, "Volume": 50}
SNAP_TOLERANCE = 0.35       # breaths whose estimate differs from the rounded value by more than step×this are not used (ambiguous)
SCALE_SMOOTH = 5            # remove outliers (e.g. two-stage inspiration) by the mode over this many neighboring breaths
# The scale derived from the Flow integral comes out about 5% smaller than actual.
# v1.3: checked against on-screen ticks on 3 videos (261 breaths): 1.06 -> 1.05 (best match when estimating without steep segments)
FLOW_GAIN = 1.05
FLOW_WINDOW = 6             # length of the segment (px) over which Volume change and Flow integral are compared
# v1.3: segments where Flow changes by more than this vs. the neighbor column (V tips, wobble) are not used for fitting (px).
# Since tips are now preserved, they no longer matched the Volume change and the estimate became noisy
FLOW_STEEP_PX = 3


def _zero_px(img, name):
    """Position of the 0 baseline (pixels from band bottom = coordinate of waveform CSV pixel values)."""
    y1, y2 = Y_RANGES[name]
    y0 = ZERO_Y[name]
    if img is not None:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        best, best_val = None, 40.0       # only gray rows brighter than the background (~30) count as the baseline
        for y in range(y0 - ZERO_SEARCH, y0 + ZERO_SEARCH + 1):
            row = gray[y, 180:880][hsv[y, 180:880, 1] < 40]   # gray pixels excluding the waveform (high saturation)
            if len(row) > 300 and np.median(row) > best_val:
                best, best_val = y, float(np.median(row))
        if best is not None:
            y0 = best
    return float(y2 - y0)


def _peak_px(values):
    """Waveform peak pixel (max excluding the top 1% outliers). None if too few points."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, 99)) if len(v) >= 20 else None


def _top_from_value(value, peak_px, zero_px, name, min_height=10):
    """Return the band top value, assuming the displayed value `value` corresponds to waveform peak pixel peak_px."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value) or value <= 0 or peak_px is None or peak_px - zero_px < min_height:
        return None
    top, bot = Y_RANGES[name]
    return value * (bot - top - zero_px) / (peak_px - zero_px)


def _flow_top_from_volume(flow_px, volume_px, volume_scale, zero_px, rate):
    """Find Flow's band top value (L/min) such that integral of Flow = change in Volume.
    Collect 'Volume change (ml)' and 'sum of Flow pixels / sweep speed' per FLOW_WINDOW px and fit a line through the origin.
    Volume stuck near 0 or at the top (end of expiration, inspiration exceeding the scale) is not used, nor are
    segments where Flow changes abruptly (FLOW_STEEP_PX)."""
    f = np.asarray(flow_px, dtype=float) - zero_px
    v = np.asarray(volume_px, dtype=float)
    vt, vb = Y_RANGES["Volume"]
    ml_per_px = (volume_scale["max"] - volume_scale["min"]) / (vb - vt)
    xs, ys = [], []
    for i in range(len(v) - FLOW_WINDOW):
        a, b = v[i], v[i + FLOW_WINDOW]
        seg = f[i:i + FLOW_WINDOW]
        if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(seg).all()):
            continue
        if min(a, b) < 4 or max(a, b) > (vb - vt) - 4:
            continue
        around = f[max(0, i - 1):i + FLOW_WINDOW + 1]
        if np.nanmax(np.abs(np.diff(around))) > FLOW_STEEP_PX:
            continue
        xs.append(seg.sum() / rate * 1000.0 / 60.0)   # ml (after multiplying by L/min/px)
        ys.append((b - a) * ml_per_px)
    if len(xs) < 30:
        return None
    xs, ys = np.array(xs), np.array(ys)
    sxx = float(np.dot(xs, xs))
    if sxx <= 0:
        return None
    lpm_per_px = float(np.dot(xs, ys)) / sxx
    ft, fb = Y_RANGES["Flow"]
    return lpm_per_px * (fb - ft - zero_px) * FLOW_GAIN


def _snap(estimate, step):
    """Round the estimate to the step. Returns None if it lies around the midpoint between steps (ambiguous)."""
    if estimate is None or not np.isfinite(estimate) or estimate <= 0:
        return None
    snapped = max(step, round(estimate / step) * step)
    return float(snapped) if abs(estimate - snapped) <= SNAP_TOLERANCE * step else None


def _smooth_tops(values, width=SCALE_SMOOTH):
    """Replace per-breath scales (None allowed) with the mode over ±width breaths. On a tie use own value; if none, the nearest breath's value."""
    from collections import Counter
    half = width // 2
    out = []
    for i, own in enumerate(values):
        window = [v for v in values[max(0, i - half):i + half + 1] if v is not None]
        if not window:
            out.append(None)
            continue
        counts = Counter(window)
        n = max(counts.values())
        tied = [v for v, c in counts.items() if c == n]
        out.append(own if own in tied else sorted(tied)[len(tied) // 2])
    # Breaths with no estimated neighbors are filled with the previous value (or the next, if none)
    last = None
    for i, v in enumerate(out):
        if v is None:
            out[i] = last
        else:
            last = v
    nxt = None
    for i in range(len(out) - 1, -1, -1):
        if out[i] is None:
            out[i] = nxt
        else:
            nxt = out[i]
    return out


def _scale_from_top(name, top_value, zero_px):
    """Build {min, max, unit} (values at band bottom/top) from the band top value and 0 baseline position."""
    top, bot = Y_RANGES[name]
    h = bot - top
    return {"min": round(-zero_px * top_value / (h - zero_px), 2), "max": top_value, "unit": SCALE_UNITS[name]}


def _estimate_scales(records, rate):
    """Compute {BreathNumber: {name: scale}} from per-breath pixel waveforms collected by BreathsToCSVs.
    records: dicts in breath order (breath, waves, zero, ppeak, vti)"""
    p_tops = [_snap(_top_from_value(r["ppeak"], _peak_px(r["waves"].get("Pressure", [])), r["zero"]["Pressure"], "Pressure"),
                    SCALE_SNAP["Pressure"]) for r in records]
    v_tops = [_snap(_top_from_value(r["vti"], _peak_px(r["waves"].get("Volume", [])), 0.0, "Volume"),
                    SCALE_SNAP["Volume"]) for r in records]
    p_tops, v_tops = _smooth_tops(p_tops), _smooth_tops(v_tops)
    f_tops = []
    for r, vt in zip(records, v_tops):
        est = None
        if vt is not None and "Flow" in r["waves"] and "Volume" in r["waves"]:
            est = _flow_top_from_volume(r["waves"]["Flow"], r["waves"]["Volume"], _scale_from_top("Volume", vt, 0.0),
                                        r["zero"]["Flow"], rate)
        f_tops.append(_snap(est, SCALE_SNAP["Flow"]))
    f_tops = _smooth_tops(f_tops)
    out = {}
    for r, pt, vt, ft in zip(records, p_tops, v_tops, f_tops):
        sc = {}
        if pt is not None:
            sc["Pressure"] = _scale_from_top("Pressure", pt, r["zero"]["Pressure"])
        if ft is not None:
            sc["Flow"] = _scale_from_top("Flow", ft, r["zero"]["Flow"])
        if vt is not None:
            sc["Volume"] = _scale_from_top("Volume", vt, 0.0)
        out[r["breath"]] = sc
    return out


def _pixels_to_physical(values, name, scale):
    """Convert pixel values to physical units by linear mapping band bottom=min, top=max."""
    top, bot = Y_RANGES[name]
    H = bot - top
    mn, mx = scale["min"], scale["max"]
    out = []
    for v in values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append(np.nan)
        else:
            # round to 2 decimals (avoids bloating CSV/JSON = slow Label rendering; still much finer than pixel resolution)
            out.append(round(mn + (float(v) / H) * (mx - mn), 2))
    return out


def _edi_scale_from_peak(edi_peak, edi_pixels, min_peak_px=8):
    """Edi's axis MAX is hard to OCR, so calibrate using the EdiPeak value (µV) already in the CSV and the waveform peak pixel.
    Edi pixels are heights from band bottom=0 (=0 µV). Peak pixel peak_px corresponds to EdiPeak.
    -> max (band top value) = EdiPeak * H / peak_px, min=0.
    Returns: {"min":0, "max":float, "unit":"uV"} / None if calibration is impossible."""
    try:
        ep = float(edi_peak)
    except (TypeError, ValueError):
        return None
    if not (ep > 0) or edi_pixels is None or len(edi_pixels) == 0:
        return None
    valid = [float(v) for v in edi_pixels if v is not None and not np.isnan(v)]
    if not valid:
        return None
    peak_px = max(valid)
    if peak_px < min_peak_px:        # calibration is unstable if amplitude is too small
        return None
    top, bot = Y_RANGES["Edi"]
    H = bot - top
    return {"min": 0, "max": round(ep * H / peak_px, 2), "unit": SCALE_UNITS["Edi"]}


def _hue_circular_dist(hue_array, target_hue):
    """Return the circular hue distance (0-90)."""
    d = np.abs(hue_array.astype(np.int16) - int(target_hue))
    return np.minimum(d, 180 - d)


# --- Hue (HSV) based waveform extraction ---
def _column_runs(img, waveform_name, screen_xs=None):
    """Per column, return vertical runs of waveform-colored pixels [(lo, hi, hue_dist)] (lo/hi are heights from band bottom).
    Pixel classification (hue, saturation, value, left-edge text exclusion, white pen tip) is the same as v1.2.
    screen_xs (0.2.0): screen x (BASE_RESOLUTION) of each column of img. The rebuilt breath strips join columns from
    both sides of a sweep wrap, so the screen position is given per column instead of x_start + column index."""
    h, w, _ = img.shape
    target_hue = TARGET_HUES.get(waveform_name)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cols = []
    for x in range(w):
        column = img[:, x, :]
        hsv_col = hsv_img[:, x, :]
        H, S, V = hsv_col[:, 0], hsv_col[:, 1], hsv_col[:, 2]
        global_x = int(screen_xs[x]) if screen_xs is not None else x + globals().get("x_start", 0)

        # Rule: for Edi, invalidate from sweep position (847) onward (same spec as v1.1)
        if waveform_name == "Edi" and 847 <= global_x:
            cols.append([])
            continue

        # Left edge (140-175) overlaps labels/text, so relax the value condition while explicitly excluding gray text
        if 140 <= global_x <= 175:
            val_min = VAL_MIN_LEFT
            text_gray_mask = (
                (column[:, 0] > 100) & (column[:, 0] < 230) &
                (np.abs(column[:, 0].astype(np.int16) - column[:, 1]) < 10) &
                (np.abs(column[:, 1].astype(np.int16) - column[:, 2]) < 10)
            )
            extra_ok = ~text_gray_mask
        else:
            val_min = VAL_MIN
            extra_ok = np.ones(h, dtype=bool)

        hue_dist = _hue_circular_dist(H, target_hue)
        valid = (hue_dist <= HUE_TOLERANCE) & (S >= SAT_MIN) & (V >= val_min) & extra_ok
        # The top rows of the band are the panel border. Excluded because misclassified as waveform color they jump to the top (=h)
        valid[:BORDER_ROWS] = False

        if not valid.any():
            # Fallback: retry with white (sweep cursor tip, and the white part of the trace at a patient trigger).
            # 0.2.0: the rows of the panel title text ("Paw cmH2O") are not used, and every vertical run of white pixels
            # is a candidate (the path choice keeps the one that continues the line). Up to 0.1.x the topmost white
            # pixel was taken, which picked the title text above a white trigger mark and drew a spike.
            if waveform_name in ["Pressure", "Flow", "Edi"]:
                white = (S <= WHITE_SAT_MAX) & (V >= WHITE_VAL_MIN) & extra_ok
                white[:max(BORDER_ROWS, TITLE_ROWS)] = False
                if white.any():
                    widx = np.where(white)[0]
                    wruns, a0 = [], widx[0]
                    for r0, r1 in zip(widx, widx[1:]):
                        if r1 > r0 + RUN_JOIN:
                            wruns.append((a0, r0))
                            a0 = r1
                    wruns.append((a0, widx[-1]))
                    cols.append([(float(h - b), float(h - a), 0.0) for a, b in wruns])
                    continue
            cols.append([])
            continue

        idx = np.where(valid)[0]
        runs = []
        a = b = idx[0]
        for r in idx[1:]:
            if r > b + RUN_JOIN:
                runs.append((a, b))
                a = r
            b = r
        runs.append((a, b))
        cols.append([(float(h - b), float(h - a), float(hue_dist[a:b + 1][valid[a:b + 1]].min()))
                     for a, b in runs])
    return cols


def _choose_runs(cols):
    """From each column's runs, select a single line connected to neighbor columns (Viterbi).
    Cost = Σ(gap to neighbor run^2 / column spacing) + RUN_HUE_WEIGHT × hue deviation.
    Split into segments where the column gap exceeds SEG_GAP. Returns: (per-column (lo, hi) or None, list of column indices per segment)"""
    chosen = [None] * len(cols)
    idx = [i for i, c in enumerate(cols) if c]
    if not idx:
        return chosen, []
    segs, cur = [], [idx[0]]
    for i in idx[1:]:
        if i - cur[-1] > SEG_GAP:
            segs.append(cur)
            cur = [i]
        else:
            cur.append(i)
    segs.append(cur)
    for seg in segs:
        cost = np.array([RUN_HUE_WEIGHT * r[2] for r in cols[seg[0]]])
        back = []
        for p, q in zip(seg, seg[1:]):
            A, B = cols[p], cols[q]
            lo_a = np.array([r[0] for r in A]); hi_a = np.array([r[1] for r in A])
            lo_b = np.array([r[0] for r in B]); hi_b = np.array([r[1] for r in B])
            gap = np.maximum(0.0, np.maximum(lo_b[:, None] - hi_a[None, :] - 1, lo_a[None, :] - hi_b[:, None] - 1))
            tr = cost[None, :] + gap * gap / (q - p)
            arg = tr.argmin(axis=1)
            back.append(arg)
            cost = tr[np.arange(len(B)), arg] + np.array([RUN_HUE_WEIGHT * r[2] for r in B])
        k = int(cost.argmin())
        chosen[seg[-1]] = cols[seg[-1]][k][:2]
        for t in range(len(seg) - 1, 0, -1):
            k = int(back[t - 1][k])
            chosen[seg[t - 1]] = cols[seg[t - 1]][k][:2]
    return chosen, segs


def _smooth_path(lo, hi, gaps):
    """Path minimizing Σ(diff to neighbor column^2 / column spacing) within each column's range [lo_i, hi_i] (Viterbi with PATH_STEP steps)."""
    grids = [np.arange(lo[i], hi[i] + 1e-9, PATH_STEP) if hi[i] > lo[i] else np.array([lo[i]])
             for i in range(len(lo))]
    cost = np.zeros(len(grids[0]))
    back = []
    for t in range(1, len(grids)):
        g0, g1 = grids[t - 1], grids[t]
        tr = cost[None, :] + (g1[:, None] - g0[None, :]) ** 2 / gaps[t - 1]
        arg = tr.argmin(axis=1)
        back.append(arg)
        cost = tr[np.arange(len(g1)), arg]
    out = np.empty(len(grids))
    k = int(cost.argmin())
    out[-1] = grids[-1][k]
    for t in range(len(grids) - 1, 0, -1):
        k = int(back[t - 1][k])
        out[t - 1] = grids[t - 1][k]
    return out


def _is_local_tip(env, sign, ext, i, j):
    """Within TIP_WIN columns left and right of tip ext at columns [i, j], there is a column receding by at least TIP_DROP,
    and no column before it protrudes beyond ext (= tip of a local peak/trough, not mid-slope)."""
    n = len(env)
    for step, start in ((-1, i - 1), (1, j + 1)):
        k = start
        ok = False
        while 0 <= k < n and abs(k - start) < TIP_WIN:
            v = sign * (env[k] - ext)
            if v > 0.5:
                return False
            if v <= -TIP_DROP:
                ok = True
                break
            k += step
        if not ok:
            return False
    return True


def _find_tips(a, b, y):
    """Return tips {position: value} where the range (bottom a / top b, adjusted to centerline) locally protrudes beyond path y."""
    n = len(y)
    pins = {}
    for sign, env in ((-1, a), (1, b)):
        out = sign * (env - y) > TIP_MARGIN
        i = 0
        while i < n:
            if not out[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and out[j + 1]:
                j += 1
            grp = np.arange(i, j + 1)
            ext = float(env[grp].max() if sign > 0 else env[grp].min())
            if _is_local_tip(env, sign, ext, i, j):
                w0, w1 = max(0, i - TIP_WIN), min(n, j + TIP_WIN + 1)
                if int((np.abs(env[w0:w1] - ext) <= TIP_SUPPORT_PX).sum()) >= TIP_SUPPORT:
                    near = grp[np.abs(env[grp] - ext) <= 0.5]
                    pins[int(near[len(near) // 2])] = ext
            i = j + 1
    return pins


# Internal function: given an image, returns waveform time-series data
def _extract_waveform(img, target_bgr, max_dist=40, waveform_name=None, y_offset=0, screen_xs=None):
    """Return the waveform time series (height from band bottom, missing = NaN) from per-column 'ranges the line passed through' (v1.3)."""
    cols = _column_runs(img, waveform_name, screen_xs)
    out = np.full(len(cols), np.nan)
    chosen, segs = _choose_runs(cols)
    if not segs:
        return list(out)
    # Line thickness (run height on gentle parts). Used to pull range ends toward the line center
    widths = np.array([c[1] - c[0] + 1 for c in chosen if c is not None])
    half = (float(np.clip(np.percentile(widths, 25), 1, 5)) - 1) / 2
    for seg in segs:
        lo = np.array([chosen[i][0] for i in seg])
        hi = np.array([chosen[i][1] for i in seg])
        a, b = lo + half, hi - half
        thin = a > b
        a[thin] = b[thin] = (lo[thin] + hi[thin]) / 2
        gaps = np.diff(seg).astype(float)
        y = _smooth_path(a, b, gaps)
        pins = _find_tips(a, b, y)
        if pins:
            a2, b2 = a.copy(), b.copy()
            for k, v in pins.items():
                a2[k] = b2[k] = v
            y = _smooth_path(a2, b2, gaps)
        out[seg] = y
    return list(out)


# --- Remove non-physiological V-shaped troughs (high->low->high) in Volume ---
# Added in v1.2:
#     When genuine waveform pixels are missed in columns with steep rise/fall, stray pixels near the band bottom (0 ml line)
#     are picked up instead, so isolated columns sink to near 0 mL (seen in Breath 5,16).
#     Volume is unimodal per breath (rise -> fall), so a high->low->high V-trough is non-physiological;
#     isolated troughs sinking relatively deep vs. both shoulders (neighboring local peaks) are treated as OCR misdetections,
#     set to NaN and restored by linear interpolation. Flow/Pressure/Edi swing both ways and are excluded.
def _reject_volume_troughs(waveforms, name="Volume", win=20,
                           trough_ratio=0.5, prom_ratio=0.25):
    v = waveforms.get(name)
    if v is None:
        return waveforms
    s = np.asarray(v, dtype=float)
    n = len(s)
    if n < 10:
        return waveforms
    filled = pd.Series(s).interpolate(method="linear", limit_direction="both").to_numpy()
    if np.isnan(filled).all():
        return waveforms
    peak = np.nanmax(filled)
    base = np.nanpercentile(filled, 5)   # near baseline
    amp = peak - base
    if amp <= 0:
        return waveforms
    bad = np.zeros(n, dtype=bool)
    for i in range(n):
        if np.isnan(s[i]):
            continue
        l0, r1 = max(0, i - win), min(n, i + win + 1)
        lmax = np.nanmax(filled[l0:i]) if i > l0 else -np.inf      # left shoulder
        rmax = np.nanmax(filled[i + 1:r1]) if i + 1 < r1 else -np.inf  # right shoulder
        shoulder = min(lmax, rmax)
        if not np.isfinite(shoulder):
            continue
        # Only troughs where both shoulders are high enough (= does not fire on monotonic rise/fall)
        # and the trough bottom sinks relatively deep vs. the shoulders, as isolated points, count as misdetections
        if (s[i] < trough_ratio * shoulder
                and shoulder - s[i] > prom_ratio * amp
                and shoulder > base + 0.1 * amp):
            bad[i] = True
    if not bad.any():
        return waveforms
    # Replace only the columns excluded as troughs with interpolated values. Pre-existing NaN gaps are
    # left untouched (deferred to downstream interpolation/missing handling) to minimize side effects.
    tmp = s.copy()
    tmp[bad] = np.nan
    interp = pd.Series(tmp).interpolate(method="linear", limit_direction="both").to_numpy()
    s_out = s.copy()
    s_out[bad] = interp[bad]
    return {**waveforms, name: s_out}


# --- Inspiration onsets (breath span) ---
# v1.3: a breath's waveform runs from its inspiration onset to the next breath's onset. The breath boundary (display
# update) comes slightly after the onset, so the onset is searched in Flow around the boundary.
# 0.2.0: the onsets directly set where each breath starts (see _BreathsToCSVs); timing is kept with Wave_T0 / Wave_Rate.
ONSET_PX = 3        # Flow above baseline by more than this is inspiratory flow (px)
ONSET_RUN = 5       # inspiratory flow must continue for at least this many columns
ONSET_PEAK = 8      # that inspiration must reach at least this far above baseline (px)
QUIET_RUN = 5       # before inspiration there must be at least this many non-inspiratory columns (excludes previous breath's inspiration remnant at the head)
ONSET_FRAC = 0.2    # from the point exceeding this fraction of the inspiratory max, trace back to the foot of the rise
EXP_PX = 3          # expiratory flow (at least this far below baseline, px)


def _inspiration_onsets(flow):
    """flow: Flow (baseline=0, up = inspiration, px, NaN allowed). List of column indices of inspiration starts (foot of the rise)."""
    raw = np.asarray(flow, dtype=float)
    g = pd.Series(raw).interpolate(limit_area="inside").to_numpy()   # missing columns are bridged only for detection
    n = len(g)
    fin = np.isfinite(g)
    ins = fin & (np.nan_to_num(g) > ONSET_PX)
    out = []
    i = 0
    while i < n:
        if not ins[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and ins[j + 1]:
            j += 1
        quiet, k = 0, i - 1
        while k >= 0 and fin[k] and not ins[k]:
            quiet += 1
            k -= 1
        peak = np.nanmax(g[i:j + 1])
        if j - i + 1 >= ONSET_RUN and peak >= ONSET_PEAK and quiet >= QUIET_RUN:
            c = i + int(np.argmax(g[i:j + 1] >= ONSET_FRAC * peak))
            while c > 0 and fin[c - 1] and g[c - 1] < g[c] - 0.25:
                c -= 1
            # When crossing a column where the rise is missing (e.g. sweep wrap seam), if just before the gap is expiration,
            # the bridged line created a fake rise, so do not treat it as an inspiration start
            k = c + int(np.argmax(np.isnan(raw[c:]))) if np.isnan(raw[c:i + 1]).any() else None
            before = raw[:k][np.isfinite(raw[:k])] if k is not None else None
            if before is None or len(before) == 0 or before[-1] >= -EXP_PX:
                out.append(c)
        i = j + 1
    return out


# --- 0.2.0: rebuilding each breath from its last image (same approach as the Hamilton engine) ---
LEAD_SEC = 1.0          # the inspiration onset is searched from this long before the breath's display update
ONSET_AFTER_SEC = 0.5   # ... to this long after it
VISIBLE_MARGIN_PX = 4   # columns behind the pen visible in one image = one sweep - erase band - this margin
MIN_BREATH_COLS = 20    # breaths shorter than this many columns (BASE_RESOLUTION) are not written
TRIG_BANDS = ("Pressure", "Flow", "Edi")   # bands whose trace turns white at a patient trigger
TRIG_WIN_SEC = 1.0      # white trigger marks are looked for from the onset to this long after it
TRIG_LEAD_COLS = 33     # ... starting this many columns (0.4 s) before the onset: a pressure trigger during expiration comes before the flow onset
TRIG_MIN_COLS = 2       # a band with at least this many columns with white pixels counts as a trigger source
TRIG_NEAR_PX = 4        # white pixels count only within this distance (rows) of the extracted trace
TRIG_TOP_ROWS = 22      # rows at the top of each band are not used (panel title text; trigger marks are at the onset, lower down)
TRIG_PEN_CLEAR = 4      # columns at the right end of a strip (white pen tip) are not used
TRIG_WHITE_VAL_MIN = 100  # brightness of a white trigger mark (thin marks are dimmer than WHITE_VAL_MIN after downscaling;
                          # the background is about 25 or less, colored traces are excluded by WHITE_SAT_MAX)
# White screen furniture (scale numbers at the bottom-left of each band, the Edi bar graph at the right end of the sweep
# area) touches the trace when it runs near it. Pixels that are white on many of the images read are not trigger marks:
# every TRIG_STATIC_EVERY-th image read goes into a map; pixels white on at least TRIG_STATIC_FRAC of them (grown by
# TRIG_STATIC_GROW px) are not used. A trigger mark stays at one place for one sweep only.
TRIG_STATIC_EVERY = 5
TRIG_STATIC_FRAC = 0.3
TRIG_STATIC_GROW = 2


def BreathsToCSVs(*args, **kwargs):
    """Call _BreathsToCSVs with frame-image caching (_frame_cache) enabled."""
    global _frame_cache
    from collections import OrderedDict
    _frame_cache = OrderedDict()
    try:
        return _BreathsToCSVs(*args, **kwargs)
    finally:
        _frame_cache = None


def _BreathsToCSVs(breaths, frames, frames_root, output_dir, user_id, overwrite=False, report_path=None, analyze_id=None, reader=None, settings=None):
    """
    Write each breath's waveforms (Pressure, Flow, Volume, Edi) to {BreathNumber}.csv and return the processed breaths.

    0.2.0 (same approach as the Hamilton engine):
    - The pen tip of every frame comes from initial_ocr (Frames.csv Sweep_Tip_Raw) and is tracked with sweep.fit_sweep
      into an unwrapped position u (columns drawn so far, BASE_RESOLUTION). Frames.csv gets Sweep_X (pen tip x in
      pixels of the analyzed video).
    - A breath runs from its inspiration onset to the next breath's onset. Its columns are read from one image: the
      last frame of the breath (the frame before the next breath's display update). That image shows every column of
      the breath that is less than one sweep (minus the erase band) behind the pen; older columns of very long breaths
      are left empty (Alert "Long"). At a wrap the Servo-U redraws the whole area (scales may change), so all columns of
      one image share that image's scale.
    - The inspiration onset is searched in Flow from LEAD_SEC before the breath's display update (the update comes
      after the onset) to ONSET_AFTER_SEC after it. Without an onset the display update time minus the typical lag is used.
    - Patient trigger: the Servo-U draws the triggering part of the waveform in white. White pixels in the Pressure /
      Flow / Edi bands from the onset to TRIG_WIN_SEC after it give Trigger ("A" patient / "C" machine) and
      Trigger_Source (bands with white marks).
    - Output rows are columns of the analyzed video (1 row = 1 screen column, ENGINE_API §7.3): the values read at
      BASE_RESOLUTION are interpolated onto those columns. Wave_T0 / Wave_Rate give the time of each row.

    Up to 0.1.x each breath was read from a start image and an end image (and the frame before each) and the
    waveform span came from their difference.
    """
    import os
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    os.makedirs(output_dir, exist_ok=True)
    if report_path is None:
        report_path = os.path.join(GS.Dir_Outputs(user_id), "BreathCSV_Report.csv")
    fps = float(GS.FPS)
    scale_enabled = (settings or {}).get("post_ocr", {}).get("scale", {}).get("enabled", True)

    frames = frames.sort_values("Frame").reset_index(drop=True)
    times = frames["Frame"].to_numpy(float) / fps
    u = frames["Sweep_U"].to_numpy(float)                       # unwrapped pen tip (BASE_RESOLUTION columns)
    speed_px_s = float(frames["Sweep_Speed"].iloc[0]) if len(frames) else sweep.SWEEP_PX_PER_SEC
    sx = float(frames["Frame_Width"].median()) / Settings.BASE_RESOLUTION[0] if "Frame_Width" in frames else 1.0
    abnormal = pd.to_numeric(frames.get("Abnormal", 0), errors="coerce").fillna(0).astype(int).to_numpy() \
        if "Abnormal" in frames else np.zeros(len(frames), int)
    seg = frames["source_file"].astype(str).str.replace("output_", "", regex=False).str.replace(".csv", "", regex=False).to_numpy()
    iframe = frames["iFrame"].astype(int).to_numpy()
    row_of_frame = {int(f): i for i, f in enumerate(frames["Frame"])}

    keep_root = os.path.join(os.path.dirname(os.path.normpath(frames_root)), KEEP_DIR_NAME)

    def path_of(row):
        """Kept image of the frame if initial_ocr kept it, else the app's frame image."""
        kept = f"{keep_root}/{seg[row]}/frame_{iframe[row]:08d}.jpg"
        return kept if os.path.isfile(kept) else f"{frames_root}/{seg[row]}/frame_{iframe[row]:08d}.jpg"

    def time_of(col):
        return np.interp(col, u, times)

    breaths = breaths.sort_values("Frame").reset_index(drop=True)
    starts = [row_of_frame.get(int(f)) for f in breaths["Frame"]]
    lead = LEAD_SEC * speed_px_s
    after = ONSET_AFTER_SEC * speed_px_s
    visible = sweep.SWEEP_W - sweep.ERASE_PX - VISIBLE_MARGIN_PX
    skip_logs, records = [], []
    skipped_alerts = {}
    static = {"sum": None, "n": 0, "reads": 0}            # white screen furniture (see TRIG_STATIC_EVERY)                                   # breaths without a waveform because of the image (Alert)

    # ---- 1. image and provisional span of each breath ----
    plans = []
    for k in range(len(breaths)):
        bn = int(breaths.at[k, "BreathNumber"])
        s = starts[k]
        e = starts[k + 1] - 1 if k + 1 < len(breaths) and starts[k + 1] is not None else None
        if k == 0 or s is None or e is None or e <= s:
            skip_logs.append({"BreathNumber": bn, "Status": "SKIPPED", "Reason": "fragment (start or end of the recording)", "Details": ""})
            plans.append(None)
            continue
        img_row = e
        while img_row < len(frames) and abnormal[img_row] and u[img_row] - visible < u[s] - lead:
            img_row += 1                                   # later frames still show the breath while it is within one sweep
        if img_row >= len(frames) or abnormal[img_row]:
            skip_logs.append({"BreathNumber": bn, "Status": "SKIPPED", "Reason": "abnormal frames", "Details": f"end image row {e}"})
            skipped_alerts[bn] = "Abnormal"
            plans.append(None)
            continue
        c_lo = max(np.floor(u[s] - lead), np.ceil(u[img_row] - visible))
        c_hi = np.floor(u[img_row])
        plans.append({"k": k, "bn": bn, "s": s, "e": e, "img": img_row, "c_lo": int(c_lo), "c_hi": int(c_hi)})

    # Open the images in ascending frame order first (the app creates them from the segment video on demand and
    # reads forward without seeking)
    for p in sorted({pl["img"] for pl in plans if pl is not None}):
        os.path.isfile(path_of(p))

    # ---- 2. read each strip, find the inspiration onset ----
    def read_strip(pl, img_row, c_from):
        """Read columns [max(c_from, oldest visible), pen] of image img_row into pl. False if the image is missing."""
        img = _load_frame(path_of(img_row))
        if img is None:
            return False
        if static["reads"] % TRIG_STATIC_EVERY == 0:
            hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            white_full = ((hsv_full[..., 1] <= WHITE_SAT_MAX) & (hsv_full[..., 2] >= TRIG_WHITE_VAL_MIN)).astype(np.uint16)
            static["sum"] = white_full if static["sum"] is None else static["sum"] + white_full
            static["n"] += 1
        static["reads"] += 1
        c_lo = int(max(np.floor(c_from), np.ceil(u[img_row] - visible)))
        c_hi = int(np.floor(u[img_row]))
        cols = np.arange(c_lo, c_hi + 1)
        xs = sweep.SWEEP_X_LEFT + np.mod(cols, sweep.SWEEP_W)
        strip = img[:, xs, :]
        waves = {}
        for name, (y1, y2) in Y_RANGES.items():
            series = pd.Series(_extract_waveform(strip[y1:y2], COLOR_DEFINITIONS[name], COLOR_TOLERANCE, waveform_name=name, screen_xs=xs))
            if IMPUTATION:
                series = series.interpolate(method="linear", limit_direction="both")
            if SMOOTHING:
                series = series.rolling(window=10, center=True, min_periods=1).mean()
            waves[name] = series.to_numpy(float)
        pl.update(img=img_row, c_lo=c_lo, c_hi=c_hi, cols=cols, xs=xs, strip=strip, waves=_reject_volume_troughs(waves),
                  zero={nm: _zero_px(img, nm) for nm in ZERO_Y})
        return True

    for pl in tqdm(plans, total=len(plans)):
        if pl is None:
            continue
        if not read_strip(pl, pl["img"], u[pl["s"]] - lead):
            skip_logs.append({"BreathNumber": pl["bn"], "Status": "SKIPPED", "Reason": "image not found", "Details": path_of(pl["img"])})
            pl["skip"] = True
            continue
        flow = pl["waves"]["Flow"] - pl["zero"]["Flow"]
        on_lo = int(np.floor(u[pl["s"]] - lead)) - pl["c_lo"]
        on_hi = int(np.ceil(u[pl["s"]] + after)) - pl["c_lo"]
        onsets = [o for o in _inspiration_onsets(flow) if on_lo <= o <= on_hi]
        pl["onset"] = (pl["c_lo"] + onsets[-1]) if onsets else None

    # Typical lag between the inspiration onset and the display update (for breaths without a detected onset)
    lags = [u[pl["s"]] - pl["onset"] for pl in plans if pl is not None and not pl.get("skip") and pl.get("onset") is not None]
    lag = float(np.median(lags)) if lags else 0.5 * speed_px_s
    live = [pl for pl in plans if pl is not None and not pl.get("skip")]
    for pl in live:
        pl["start_col"] = pl["onset"] if (TRIM_TO_BREATH and pl["onset"] is not None) else int(round(u[pl["s"]] - (lag if TRIM_TO_BREATH else 0)))
    for a, b in zip(live, live[1:]):
        a["end_col"] = b["start_col"] if b["k"] == a["k"] + 1 else int(np.floor(u[a["e"]]))
    if live:
        live[-1]["end_col"] = int(np.floor(u[live[-1]["e"]]))

    # The next breath's onset can lie beyond the pen of this breath's last image (onset after the display update):
    # read the breath again from the first later image whose pen has passed it
    rereads = []
    for pl in live:
        if pl["end_col"] - 1 <= pl["c_hi"]:
            continue
        r = pl["img"] + 1
        while r < len(frames) and (u[r] < pl["end_col"] - 1 or abnormal[r]) and u[r] - visible <= pl["start_col"]:
            r += 1
        if r < len(frames) and u[r] >= pl["end_col"] - 1 and not abnormal[r] and u[r] - visible <= pl["start_col"]:
            rereads.append((r, pl))
    for r in sorted({r for r, _ in rereads}):               # open in ascending order first (no seeking)
        os.path.isfile(path_of(r))
    for r, pl in rereads:
        read_strip(pl, r, pl["start_col"] - TRIG_LEAD_COLS)

    # ---- 3. cut each breath to [start_col, end_col), trigger marks, records for the scale estimate ----
    static_white = None
    if static["n"]:
        static_white = (static["sum"] >= TRIG_STATIC_FRAC * static["n"]).astype(np.uint8)
        if TRIG_STATIC_GROW:
            static_white = cv2.dilate(static_white, np.ones((2 * TRIG_STATIC_GROW + 1,) * 2, np.uint8))
        static_white = static_white.astype(bool)
    for pl in live:
        c0 = max(pl["start_col"], pl["c_lo"])
        c1 = min(pl["end_col"], pl["c_hi"] + 1)
        alerts = []
        if pl["start_col"] < pl["c_lo"]:
            alerts.append("Long")                          # older than one sweep: not visible in the image
        if pl["end_col"] > pl["c_hi"] + 1:
            alerts.append("Partial")
        if c1 - c0 < MIN_BREATH_COLS:
            skip_logs.append({"BreathNumber": pl["bn"], "Status": "SKIPPED", "Reason": "too short", "Details": f"{c1 - c0} columns"})
            continue
        i0, i1 = c0 - pl["c_lo"], c1 - pl["c_lo"]
        waves = {nm: v[i0:i1] for nm, v in pl["waves"].items()}
        # the pen tip at the right end of the strip is also drawn in white: keep clear of it
        trig_hi = min(i1, i0 + int(round(TRIG_WIN_SEC * speed_px_s)), len(pl["cols"]) - TRIG_PEN_CLEAR)
        marks = _trigger_marks(pl["strip"], pl["waves"], max(0, i0 - TRIG_LEAD_COLS), trig_hi,
                               static=static_white[:, pl["xs"]] if static_white is not None else None)
        records.append({
            "breath": pl["bn"], "k": pl["k"], "c0": c0, "c1": c1, "waves": waves, "zero": pl["zero"],
            "ppeak": breaths.at[pl["k"], "Real_Ppeak"] if "Real_Ppeak" in breaths else np.nan,
            "vti": breaths.at[pl["k"], "Real_VTi"] if "Real_VTi" in breaths else np.nan,
            "edi_peak": breaths.at[pl["k"], "EdiPeak"] if "EdiPeak" in breaths else np.nan,
            "edi_full": waves["Edi"], "marks": marks, "alerts": alerts,
            "start_name": path_of(pl["s"]), "end_name": path_of(pl["img"]),
        })
        pl.pop("strip", None)

    # ---- 4. axis scales, physical units, output on the analyzed video's columns ----
    estimated = _estimate_scales(records, speed_px_s) if scale_enabled else {}
    last_edi_scale = None
    out_cols = {c: {} for c in ("P_min", "P_max", "F_min", "F_max", "V_min", "V_max", "E_min", "E_max", "Wave_T0", "Wave_Rate",
                                "Wave_Points", "Trigger", "Trigger_Source", "Trig_White_P", "Trig_White_F", "Trig_White_E",
                                "Start_File_Name", "End_File_Name", "Alert_Wave")}
    written = 0
    for rec in records:
        bn = rec["breath"]
        scale = dict(estimated.get(bn, {})) if scale_enabled else {}
        if scale_enabled:
            edi_scale = _edi_scale_from_peak(rec["edi_peak"], rec["edi_full"])
            if edi_scale:
                scale["Edi"] = last_edi_scale = edi_scale
            elif last_edi_scale is not None:
                scale["Edi"] = last_edi_scale
        base_cols = np.arange(rec["c0"], rec["c1"], dtype=float)
        n0, n1 = int(np.ceil(rec["c0"] * sx)), int(np.ceil(rec["c1"] * sx))
        native = np.arange(n0, n1)
        df_out = pd.DataFrame({"X": native})
        for nm in ("Pressure", "Flow", "Volume", "Edi"):
            vals = _resample(base_cols, rec["waves"][nm], native / sx)
            df_out[nm] = _pixels_to_physical(vals.tolist(), nm, scale[nm]) if nm in scale else np.round(vals, 2)
        path = os.path.join(output_dir, f"{bn}.csv")
        if overwrite or not os.path.exists(path):
            df_out.to_csv(path, index=False)
        written += 1
        skip_logs.append({"BreathNumber": bn, "Status": "OK", "Reason": "", "Details": ""})
        short = {"Pressure": "P", "Flow": "F", "Volume": "V", "Edi": "E"}
        for nm, pfx in short.items():
            if nm in scale:
                out_cols[f"{pfx}_min"][bn], out_cols[f"{pfx}_max"][bn] = scale[nm]["min"], scale[nm]["max"]
        out_cols["Wave_T0"][bn] = round(float(time_of(n0 / sx)), 4)
        out_cols["Wave_Rate"][bn] = round(speed_px_s * sx, 4)
        out_cols["Wave_Points"][bn] = len(native)
        src = [nm for nm, key in (("Pressure", "P"), ("Flow", "F"), ("Edi", "E")) if rec["marks"][nm] >= TRIG_MIN_COLS]
        out_cols["Trigger"][bn] = "A" if src else "C"
        out_cols["Trigger_Source"][bn] = "+".join(src)
        for nm, key in (("Pressure", "P"), ("Flow", "F"), ("Edi", "E")):
            out_cols[f"Trig_White_{key}"][bn] = rec["marks"][nm]
        out_cols["Start_File_Name"][bn], out_cols["End_File_Name"][bn] = rec["start_name"], rec["end_name"]
        out_cols["Alert_Wave"][bn] = ",".join(rec["alerts"])

    for bn, a in skipped_alerts.items():
        out_cols["Alert_Wave"][bn] = a
    for col, values in out_cols.items():
        breaths[col] = breaths["BreathNumber"].map(values)
    # Waveform alerts join the breath alerts (comma separated)
    alert = breaths.get("Alert", pd.Series("", index=breaths.index)).fillna("").astype(str)
    breaths["Alert"] = [",".join(dict.fromkeys([x for x in (a.split(",") + w.split(",")) if x and x != "nan"]))
                        for a, w in zip(alert, breaths["Alert_Wave"].fillna("").astype(str))]
    breaths = breaths.drop(columns=["Alert_Wave"])
    report_df = pd.DataFrame(skip_logs, columns=["BreathNumber", "Status", "Reason", "Details"]).sort_values("BreathNumber")
    report_df.to_csv(report_path, index=False)
    return written, breaths


def _resample(src_cols, values, dst_cols):
    """Linear interpolation of values (at src_cols) onto dst_cols; NaN where either neighbouring sample is NaN."""
    v = np.asarray(values, float)
    out = np.interp(dst_cols, src_cols, np.nan_to_num(v))
    j = np.clip(np.searchsorted(src_cols, dst_cols, side="right") - 1, 0, len(src_cols) - 1)
    j2 = np.clip(j + 1, 0, len(src_cols) - 1)
    out[~(np.isfinite(v[j]) & np.isfinite(v[j2]))] = np.nan
    return out


def _trigger_marks(strip, waves, c_from, c_to, static=None):
    """Number of columns in [c_from, c_to) where the trace itself is white (patient trigger), per band (Pressure, Flow, Edi).
    Only pixels within TRIG_NEAR_PX of the extracted trace count, so panel titles, scale numbers and the Edi bar graph
    (where the trace is empty) are not mistaken for trigger marks. static: white screen furniture for the strip's columns
    (same shape as the strip's rows x columns); those pixels do not count."""
    if c_to <= c_from:
        return {nm: 0 for nm in TRIG_BANDS}
    hsv = cv2.cvtColor(strip[:, c_from:c_to], cv2.COLOR_BGR2HSV)
    white_all = (hsv[..., 1] <= WHITE_SAT_MAX) & (hsv[..., 2] >= TRIG_WHITE_VAL_MIN)
    if static is not None:
        white_all &= ~static[:, c_from:c_to]
    out = {}
    for nm in TRIG_BANDS:
        y1, y2 = Y_RANGES[nm]
        top = y1 + TRIG_TOP_ROWS
        trace = np.asarray(waves[nm], float)[c_from:c_to]           # height from the band bottom
        n = 0
        for i, hgt in enumerate(trace):
            if not np.isfinite(hgt):
                continue
            row = int(round(y2 - hgt))
            if white_all[max(top, row - TRIG_NEAR_PX):min(y2, row + TRIG_NEAR_PX + 1), i].any():
                n += 1
        out[nm] = n
    return out


class AbnormalFrameDetector:
    """v1.4: abnormal frame (e.g. popup) detection is done inside initial_ocr, on frames in memory.

    Up to v1.3, post-processing DetectAbnormalFrames re-read every frame's JPEG (36,000 images for a 1-hour video).
    The criterion (compare ROI_LIST_Abnormal_Detection areas to a reference; abnormal if mean diff, std diff or histogram correlation
    crosses its threshold) and the thresholds are the same as v1.3.

    Reference: in v1.3 it was the first 'non-black' frame of the video. initial_ocr runs per segment (default 60 s), so
    use the per-pixel median of the segment's non-black frames (so the reference holds even if the segment starts with a popup).

    Usage: call add(frame) in the same order as initial_ocr rows, and finally get
    [(Abnormal, MeanDiff, StdDiff, HistCorr), ...] from results().
    """

    THRESH_MEAN = 30         # mean-diff threshold (large = abnormal)
    THRESH_STD = 10          # std diff (large = abnormal)
    THRESH_HIST_CORR = 0.90  # histogram correlation (small = abnormal)
    BLACK_MEAN = 20          # frames whose overall mean is at or below this are black screens (not used as reference)
    MAX_REF_FRAMES = 200     # max number of frames for the reference median (evenly subsampled)

    def __init__(self, roi_list=None):
        self.roi_list = list(roi_list or Settings.ROI_LIST_Abnormal_Detection)
        self.strips = []     # per frame: [grayscale image per ROI]
        self.black = []      # per frame: whether it is a black screen

    def add(self, bgr):
        """Add one color frame normalized to the base resolution (Settings.BASE_RESOLUTION)."""
        self.strips.append([cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                            for (x1, y1, x2, y2) in self.roi_list])
        self.black.append(bool(bgr.mean() <= self.BLACK_MEAN))

    @staticmethod
    def _hist(gray):
        h = cv2.calcHist([gray], [0], None, [64], [0, 256])
        return cv2.normalize(h, h).flatten()

    def results(self):
        n = len(self.strips)
        normal = [i for i in range(n) if not self.black[i]]
        if not normal:
            # Segment is all black (or empty): waveform unreadable, so mark abnormal
            return [(1, 0.0, 0.0, 0.0)] * n
        step = max(1, len(normal) // self.MAX_REF_FRAMES)
        sample = normal[::step]
        refs = [np.median(np.stack([self.strips[i][k] for i in sample]), axis=0).astype(np.uint8)
                for k in range(len(self.roi_list))]
        ref_stds = [np.std(r) for r in refs]
        ref_hists = [self._hist(r) for r in refs]

        out = []
        for grays in self.strips:
            abnormal = 0
            mean_diff = std_diff = hist_corr = 0.0
            for k, gray in enumerate(grays):
                mean_diff = float(np.mean(cv2.absdiff(refs[k], gray)))
                std_diff = float(abs(np.std(gray) - ref_stds[k]))
                hist_corr = float(cv2.compareHist(ref_hists[k], self._hist(gray), cv2.HISTCMP_CORREL))
                if (mean_diff > self.THRESH_MEAN or std_diff > self.THRESH_STD
                        or hist_corr < self.THRESH_HIST_CORR):
                    abnormal = 1
                    break  # exit early if abnormal by either
            out.append((abnormal, mean_diff, std_diff, hist_corr))
        return out
