import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import Global_Setting as GS
from . import Settings

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
    if (img.shape[1], img.shape[0]) != Settings.BASE_RESOLUTION:
        interp = cv2.INTER_AREA if img.shape[1] > Settings.BASE_RESOLUTION[0] else cv2.INTER_LINEAR
        img = cv2.resize(img, Settings.BASE_RESOLUTION, interpolation=interp)
    if _frame_cache is not None:
        _frame_cache[path] = img.copy()
        while len(_frame_cache) > FRAME_CACHE_SIZE:
            _frame_cache.popitem(last=False)
    return img


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
SWEEP_RATE_DEFAULT = 81.6   # sweep speed px/s (81.5-81.7 on 3 videos). Used when it cannot be derived from the sweep positions in start/end images
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


def _scale_frame_path(frames_root, source_file, iframe):
    sub = str(source_file).replace("output_", "").replace(".csv", "")
    return os.path.join(frames_root, sub, f"frame_{int(iframe):08d}.jpg")


def _hue_circular_dist(hue_array, target_hue):
    """Return the circular hue distance (0-90)."""
    d = np.abs(hue_array.astype(np.int16) - int(target_hue))
    return np.minimum(d, 180 - d)


# --- Hue (HSV) based waveform extraction ---
def _column_runs(img, waveform_name):
    """Per column, return vertical runs of waveform-colored pixels [(lo, hi, hue_dist)] (lo/hi are heights from band bottom).
    Pixel classification (hue, saturation, value, left-edge text exclusion, white pen tip) is the same as v1.2."""
    h, w, _ = img.shape
    target_hue = TARGET_HUES.get(waveform_name)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cols = []
    for x in range(w):
        column = img[:, x, :]
        hsv_col = hsv_img[:, x, :]
        H, S, V = hsv_col[:, 0], hsv_col[:, 1], hsv_col[:, 2]
        global_x = x + globals().get("x_start", 0)

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
            # Fallback: retry with white (sweep cursor tip)
            if waveform_name in ["Pressure", "Flow", "Edi"]:
                white = (S <= WHITE_SAT_MAX) & (V >= WHITE_VAL_MIN) & extra_ok
                white[:BORDER_ROWS] = False
                if white.any():
                    r = int(np.where(white)[0][0])
                    cols.append([(float(h - r), float(h - r), 0.0)])
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
def _extract_waveform(img, target_bgr, max_dist=40, waveform_name=None, y_offset=0):
    """Return the waveform time series (height from band bottom, missing = NaN) from per-column 'ranges the line passed through' (v1.3)."""
    cols = _column_runs(img, waveform_name)
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

# --- Extract time-series data from the image ---
def extract_waveform_timeseries(image_path, x_start=524, x_end=807):
    globals()["x_start"] = x_start
    img = _load_frame(image_path)
    waveforms = {}
    for name, (y1, y2) in Y_RANGES.items():
        crop = img[y1:y2, x_start:x_end]
        ts = _extract_waveform(crop, COLOR_DEFINITIONS[name], COLOR_TOLERANCE, waveform_name=name, y_offset=y1)
        series = pd.Series(ts)
        if IMPUTATION:
            series = pd.Series(ts).interpolate(method='linear', limit_direction='both')
        if SMOOTHING:
            # Up to v1.2 the moving-average result was immediately overwritten, so smoothing had no effect (fixed in v1.3).
            # Default is OFF. Turning it ON erodes V-shaped tips
            series = series.rolling(window=10, center=True, min_periods=1).mean()
        waveforms[name] = series.to_numpy()
    return waveforms

# --- Difference region detection ---
import cv2
import numpy as np

def get_regions_for_y_range(img1, img2, y_start, y_end,
                             x_start, x_end,
                             diff_thresh=25, smooth_kernel=15, min_gap=30):
    gray1 = cv2.cvtColor(img1[y_start:y_end, x_start:x_end], cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2[y_start:y_end, x_start:x_end], cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    column_max_diff = np.max(diff, axis=0)
    smoothed = np.convolve(column_max_diff, np.ones(smooth_kernel)/smooth_kernel, mode='same')
    x_coords = np.where(smoothed > diff_thresh)[0]

    if len(x_coords) == 0:
        return []

    regions = []
    start = x_coords[0]
    for i in range(1, len(x_coords)):
        if x_coords[i] > x_coords[i - 1] + min_gap:
            regions.append((start + x_start, x_coords[i - 1] + x_start))
            start = x_coords[i]
    regions.append((start + x_start, x_coords[-1] + x_start))
    return regions


def get_difference_regions(image_path1, image_path2):
    x_start, x_end = 140, 893

    # Y ranges (3 panels: Pressure/Flow/Volume). References Y_RANGES as the single source of truth
    # (previously separate values were hardcoded here, a breeding ground for drift).
    y_ranges = [Y_RANGES["Pressure"], Y_RANGES["Flow"], Y_RANGES["Volume"]]

    # Load images
    img1 = _load_frame(image_path1)
    img2 = _load_frame(image_path2)

    # Extract difference regions for each Y range
    region_sets = []
    for y_start, y_end in y_ranges:
        regions = get_regions_for_y_range(img1, img2, y_start, y_end, x_start, x_end)
        region_sets.append(regions)

    # Prepare a bitmask to check overlap along the X axis
    x_length = x_end + 1  # supports up to 893
    mask = np.zeros(x_length, dtype=int)

    # Count overlaps for each region set
    for regions in region_sets:
        temp = np.zeros(x_length, dtype=int)
        for x1, x2 in regions:
            temp[x1:x2 + 1] = 1
        mask += temp

    # Get X positions overlapped by 2 or more regions
    overlap_coords = np.where(mask >= 2)[0]
    if len(overlap_coords) == 0:
        return []

    # Convert X coordinates back to regions
    merged = []
    start = overlap_coords[0]
    for i in range(1, len(overlap_coords)):
        if overlap_coords[i] > overlap_coords[i - 1] + 1:
            merged.append((start, overlap_coords[i - 1]))
            start = overlap_coords[i]
    merged.append((start, overlap_coords[-1]))

    return merged


# --- Concatenate waveforms of difference regions ---
def merge_waveforms_across_regions(image_path, regions):
    merged = {}
    for x_start, x_end in sorted(regions, key=lambda r: -r[0]):
        waveforms = extract_waveform_timeseries(image_path, x_start, x_end)
        for key, values in waveforms.items():
            merged.setdefault(key, []).extend(values)
    return {k: np.array(v) for k, v in merged.items()}


# --- Detect sweep cursor (tip) in the end frame ---
# Added in v1.2:
#     The start/end 2-frame diff, for long-interval breaths, picked up not only the current sweep but also the
#     'old waveform (afterimage) of the previous sweep remaining to the right of the end cursor', looking like multiple waveforms.
#     The diff band between the end frame and the frame just before it = sweep tip (cursor) position; columns to its right
#     are previous-sweep afterimage, so return the reference x used to drop them from the extraction regions.
def _sweep_cursor_from_imgs(img_e, img_b, x_lo=140, x_hi=893,
                            diff_thresh=25, smooth_kernel=15,
                            vote_panels=("Pressure", "Flow", "Volume"),
                            min_vote=2, cluster_gap=10, wrap_width=250):
    """Return sweep cursor x (= tip = right edge of new drawing) from 2 images (end / previous, both base resolution).
    None if not detectable. Images are assumed already loaded via _load_frame (BASE_RESOLUTION).

    The cursor is a vertical feature spanning all waveform panels, so in each Pressure/Flow/Volume panel
    vote per column on whether the smoothed diff exceeds the threshold, and adopt only columns with >= min_vote(=2) votes
    (same majority vote as get_difference_regions; Edi is excluded since its panel-specific marker corrupts single-panel results).
    This allows full-width search up to x_hi without being fooled by the Edi marker or single-panel noise.
    Adopted columns are clustered; if wide clusters (sweep wrap) exist, return the left edge of the rightmost wide cluster,
    otherwise the right edge (= tip) of the rightmost cluster."""
    if img_e is None or img_b is None:
        return None
    width = x_hi - x_lo
    vote = np.zeros(width, dtype=np.int16)
    kernel = np.ones(smooth_kernel) / smooth_kernel
    for nm in vote_panels:
        y0, y1 = Y_RANGES[nm]
        g_e = cv2.cvtColor(img_e[y0:y1, x_lo:x_hi], cv2.COLOR_BGR2GRAY).astype(np.int16)
        g_b = cv2.cvtColor(img_b[y0:y1, x_lo:x_hi], cv2.COLOR_BGR2GRAY).astype(np.int16)
        col = np.max(np.abs(g_e - g_b), axis=0)
        sm = np.convolve(col, kernel, mode='same')
        vote += (sm > diff_thresh)
    xs = np.where(vote >= min_vote)[0]
    if len(xs) == 0:
        return None
    # Cluster adopted columns (columns within cluster_gap are grouped into one cluster)
    runs = []
    start = prev = xs[0]
    for x in xs[1:]:
        if x > prev + cluster_gap:
            runs.append((start, prev))
            start = x
        prev = x
    runs.append((start, prev))
    # Sweep wrap: if a wide cluster exists, the left edge of the rightmost wide cluster = cursor
    wide = [r for r in runs if (r[1] - r[0]) > wrap_width]
    if wide:
        return x_lo + int(wide[-1][0])
    # Normal: right edge of the rightmost cluster = sweep tip
    return x_lo + int(runs[-1][1])


def _detect_sweep_cursor(path_end, path_before, x_lo=140, x_hi=893,
                         diff_thresh=25, smooth_kernel=15):
    """Return the end frame's sweep cursor x (= tip = rightmost newly drawn column). None if not detectable."""
    return _sweep_cursor_from_imgs(
        _load_frame(path_end), _load_frame(path_before),
        x_lo=x_lo, x_hi=x_hi, diff_thresh=diff_thresh, smooth_kernel=smooth_kernel,
    )


def annotate_sweep_lines(df_frames, frames_root, line_bgr=(0, 0, 255), thickness=2):
    """Draw the sweep cursor (Sweep Line) as a vertical line on each frame and overwrite the original frame image.
    Detection runs at base resolution (BASE_RESOLUTION); drawing scales coordinates to the original-resolution frame
    (keeps original resolution = consistent with Label's original-resolution download). Detects tip x from the diff with
    the previous (unmodified) frame and draws a line spanning the waveform band (Pressure top to Edi bottom). Diagnostic for when the
    Detect Sweep Line option (default OFF) is enabled. Undetectable frames are left unmarked. Returns the number of frames drawn."""
    if df_frames is None or getattr(df_frames, "empty", True):
        return 0
    base_w, base_h = Settings.BASE_RESOLUTION
    y0 = int(Y_RANGES["Pressure"][0])
    y1 = int(Y_RANGES["Edi"][1])
    ordered = df_frames.sort_values("Frame")
    prev_norm = None   # base-resolution image of previous frame (detection only, unmodified)
    drawn = 0
    for _, fr in ordered.iterrows():
        path = _scale_frame_path(frames_root, fr.get("source_file"), fr.get("iFrame"))
        norm = _load_frame(path)               # base resolution (for detection)
        if norm is None:
            prev_norm = None
            continue
        cursor_x = _sweep_cursor_from_imgs(norm, prev_norm) if prev_norm is not None else None
        prev_norm = norm                       # kept unmodified for next frame's detection (no line contamination)
        if cursor_x is None:
            continue
        orig = cv2.imread(path)                # original resolution (draw and overwrite)
        if orig is None:
            continue
        oh, ow = orig.shape[:2]
        sx = ow / base_w
        sy = oh / base_h
        x = int(round(cursor_x * sx))
        ya = int(round(y0 * sy))
        yb = int(round(y1 * sy))
        cv2.line(orig, (x, ya), (x, yb), line_bgr, max(1, int(round(thickness * sx))))
        cv2.imwrite(path, orig)
        drawn += 1
    return drawn                  # tip = rightmost changed column


def _trim_regions_right_of_cursor(regions, cursor_x, left_edge=160, right_edge=872):
    """Drop regions right of the end cursor (= previous-sweep afterimage).
    However, for a true wrap (separate regions at both left and right edges), the right side is this breath's genuine drawing,
    so do not trim and defer to the existing processing."""
    if cursor_x is None or not regions:
        return regions
    right_touch = any(r[1] >= right_edge for r in regions)
    left_touch = any(r[0] <= left_edge for r in regions)
    is_true_wrap = right_touch and left_touch and len(regions) >= 2
    if is_true_wrap:
        return regions
    trimmed = [(xs, min(xe, cursor_x)) for xs, xe in regions if xs <= cursor_x]
    return [(xs, xe) for xs, xe in trimmed if xe > xs]


# --- Breath region reconstruction for sweep wrap ---
# Added in v1.2:
#     The diff picks up every X position where 'only one of the start/end frames has a waveform', so for breaths where the sweep
#     advances more than one breath / wraps at the right edge, old afterimage fragments of other breaths got mixed in, and concatenation
#     looked like two humps (plateaus).
#     When a wrap is detected, adopt only 'right-edge region (drawn just before wrap) -> left-edge region (drawn just after wrap)'
#     in time order and discard old afterimage fragments in between (non-wrap cases use all regions as before).
def _select_breath_regions(regions, left_edge=160, right_edge=872):
    if regions is None or len(regions) <= 1:
        return regions
    right = [r for r in regions if r[1] >= right_edge]   # regions touching the right edge of the screen
    left  = [r for r in regions if r[0] <= left_edge]    # regions touching the left edge of the screen
    if right and left:
        # Wrap detected: adopt only right-edge -> left-edge breath fragments, discard afterimages in between
        sel = sorted(right, key=lambda r: r[0]) + sorted(left, key=lambda r: r[0])
        seen, out = set(), []
        for r in sel:
            key = (int(r[0]), int(r[1]))
            if key not in seen:
                seen.add(key)
                out.append(r)
        return out
    # Non-wrap: as before (diff is local, afterimage contamination unlikely)
    return regions


# --- Trim the 'next breath rise' mixed in at the end ---
# Added in v1.2:
#     On wrap, the left-edge region's diff can extend to the start of the next breath's inspiratory rise,
#     leaving a small part of the next breath's rise at the end of the reconstructed waveform. In the Pressure waveform, detect the point
#     where it 'returns to baseline once and then rises again (= next breath's inspiration)' and cut all waveforms from there on.
#     A breath is normally baseline -> inspiration -> plateau -> expiration -> baseline with no second rise,
#     so trim conservatively only when a second rise is detected.
def _trim_trailing_breath(waveforms, ref="Pressure"):
    P = waveforms.get(ref)
    if P is None:
        return waveforms
    P = np.asarray(P, dtype=float)
    n = len(P)
    if n < 60:
        return waveforms
    s = pd.Series(P).interpolate(method="linear", limit_direction="both").to_numpy()
    if np.isnan(s).all():
        return waveforms
    base = np.nanpercentile(s, 20)
    peak = np.nanpercentile(s, 90)
    amp = peak - base
    if amp < 15:  # small amplitude (flat/missing) -> undeterminable, do not trim
        return waveforms
    hi = base + 0.60 * amp
    lo = base + 0.30 * amp
    above = s > hi
    if not above.any():
        return waveforms
    first_up = int(np.argmax(above))                       # first inspiration
    after = np.where(s[first_up:] < lo)[0]                  # then returns near baseline
    if len(after) == 0:
        return waveforms
    back = first_up + int(after[0])
    nxt = np.where(s[back:] > hi)[0]                        # rises again (next breath's inspiration)
    if len(nxt) == 0:
        return waveforms
    cut = back + int(nxt[0])
    # Move back to the rise onset (last point near baseline just before cut) and cut there
    pre = np.where(s[back:cut] <= lo)[0]
    if len(pre) > 0:
        cut = back + int(pre[-1]) + 1
    # Guard against cutting too early (mistakenly cutting the main breath)
    if cut < n * 0.5:
        return waveforms
    return {k: np.asarray(v)[:cut] for k, v in waveforms.items()}

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


# --- Trim to the breath span ---
# Added in v1.3:
#     The breath boundary (Cdyn display update) is slightly offset from inspiration start, so the start image is taken 0.6 s
#     before the boundary (Frame-6) and the end image up to the next breath's boundary. As a result, the waveform CSV started with the
#     previous breath's end of expiration (about 0.4-0.6 s) and ended with the next breath's inspiratory rise.
#     Find Flow inspiratory rises; trim the head to the first inspiration start and the tail to the start of the 'last inspiration
#     without expiration' (= the next breath's inspiration cut off midway). Additionally, cut the previous breath's tail at the
#     inspiration start time found in the next breath (to also remove rises too short to be classified as inspiration on their own).
#     Timing of the trimmed part is preserved via Breaths.csv Wave_T0 (time of first point, s) / Wave_Rate (px/s).
#     Computation is just a 1-D scan of Flow per breath.
ONSET_PX = 3        # Flow above baseline by more than this is inspiratory flow (px)
ONSET_RUN = 5       # inspiratory flow must continue for at least this many columns
ONSET_PEAK = 8      # that inspiration must reach at least this far above baseline (px)
QUIET_RUN = 5       # before inspiration there must be at least this many non-inspiratory columns (excludes previous breath's inspiration remnant at the head)
ONSET_FRAC = 0.2    # from the point exceeding this fraction of the inspiratory max, trace back to the foot of the rise
EXP_PX = 3          # expiratory flow (at least this far below baseline, px)
TRIM_LEAD = 3       # how many columns before inspiration start to keep


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


def _breath_trim_range(flow, valid_rows):
    """Return the range of rows to keep (head, tail) and the first inspiration start (None if none).
    valid_rows: rows where any waveform has a value (empty leading/trailing rows are also trimmed)."""
    n = len(flow)
    if valid_rows.any():
        lo, hi = int(np.argmax(valid_rows)), n - int(np.argmax(valid_rows[::-1]))
    else:
        lo, hi = 0, n
    onsets = _inspiration_onsets(flow)
    if not onsets:
        return lo, hi, None
    head = max(lo, onsets[0] - TRIM_LEAD)
    tail = hi
    last = onsets[-1]
    # No expiration after the last inspiration = next breath's inspiration is cut off and included (in a breath containing 2 breaths, keep the second)
    if last != onsets[0] and not (np.nan_to_num(np.asarray(flow[last:], float)) < -EXP_PX).any():
        tail = max(head + 1, min(hi, last - TRIM_LEAD))
    return head, tail, onsets[0]


def _first_sample_time(start_frame, x0, cursor_start, rate, offset):
    """Time (s) of the first waveform point. Same method as the app's ResultPackage.sample_times:
    back-calculated from the start image's pen tip (sweep position - erase band width offset). Breaths containing a wrap use the start image time."""
    ts = float(start_frame) / GS.FPS
    if cursor_start is not None and x0 is not None and 0 < cursor_start \
            and 0 <= cursor_start - x0 <= max(200, 3 * offset):
        return ts - (cursor_start - offset - x0) / rate
    return ts


def _trim_records_to_breaths(records, rate):
    """Trim records (breaths in time order). Writes head / tail / wave_t0 into each rec."""
    offsets = [r["cursor_start"] - r["x0"] for r in records
               if r["cursor_start"] is not None and r["x0"] is not None and 0 <= r["cursor_start"] - r["x0"] <= 200]
    offset = float(np.median(offsets)) if offsets else 0.0
    for r in records:
        flow = r["waves"]["Flow"] - r["zero"]["Flow"]
        valid = np.isfinite(np.vstack([r["waves"][nm] for nm in ("Pressure", "Flow", "Volume", "Edi")])).any(axis=0)
        r["t_first"] = _first_sample_time(r["start_frame"], r["x0"], r["cursor_start"], rate, offset)
        if TRIM_TO_BREATH:
            r["head"], r["tail"], onset = _breath_trim_range(flow, valid)
        else:
            r["head"], r["tail"], onset = 0, len(flow), None
        r["onset_time"] = None if onset is None else r["t_first"] + onset / rate
    if TRIM_TO_BREATH:
        # Remove from the previous breath's tail anything after the next breath's inspiration start
        for a, b in zip(records, records[1:]):
            # Breaths without their own inspiration start (broken breaths) have dubious timing, so do not cut
            if a["onset_time"] is None or b["onset_time"] is None or b["onset_time"] <= a["onset_time"]:
                continue
            cut = int(np.ceil((b["onset_time"] - a["t_first"]) * rate)) - TRIM_LEAD
            if a["head"] + 1 < cut < a["tail"]:
                a["tail"] = cut
    for r in records:
        r["wave_t0"] = r["t_first"] + r["head"] / rate


# --- Main function: generate per-breath waveform CSVs ---
def AllWaveformTocsv(waves_csv="./outputs/FinalBreath.csv", frames_root="./frames", output_dir="./Breath"):
    os.makedirs(output_dir, exist_ok=True)

    breaths = pd.read_csv(waves_csv)
    breaths["Start_Frame"] = breaths["iFrame"]
    breaths["End_Frame"] = breaths["iFrame"].shift(-1) 
    breaths = breaths.fillna(-1).astype({"Start_Frame": int, "End_Frame": int})

    all_records = []
    for _, row in tqdm(breaths.iterrows(), total=len(breaths)):
        folder_id = str(row["source_file"]).split("_")[-1].split(".")[0]
        folder_path = os.path.join(frames_root, folder_id)
        start, end = row["Start_Frame"], row["End_Frame"]
        if start < 0 or end <= start:
            continue
        img_start = os.path.join(folder_path, f"frame_{start:08d}.jpg")
        img_end = os.path.join(folder_path, f"frame_{end:08d}.jpg")
        if not (os.path.isfile(img_start) and os.path.isfile(img_end)):
            continue
        regions = get_difference_regions(img_end, img_start)
        if not regions:
            continue
        regions = _select_breath_regions(regions)   # remove afterimage fragments on sweep wrap
        waveforms = merge_waveforms_across_regions(img_end, regions)
        waveforms = _trim_trailing_breath(waveforms) # trim the next breath's rise at the end
        waveforms = _reject_volume_troughs(waveforms) # remove non-physiological V-troughs in Volume
        x_start = regions[-1][0]
        for name, series in waveforms.items():
            for i, val in enumerate(series):
                all_records.append({
                    "BreathNumber": row["BreathNumber"],
                    "WaveType": name,
                    "X": x_start + i,
                    "Value": val
                })
    out_df = pd.DataFrame(all_records)
    out_df["X"] = out_df.groupby(["BreathNumber", "WaveType"]).cumcount() + 1
    out_df.to_csv(os.path.join(output_dir, "Waves.csv"), index=False)
    print("Waves.csv generated.")

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
    Extract difference regions from each breath episode's Start/End interval and build waveforms (Pressure, Flow, Volume, Edi).
    Writes {BreathNumber}.csv and returns the processed breaths.

    Additional options:
        overwrite (bool): if True, delete existing {BreathNumber}.csv and overwrite
        report_path (str|None): where to save the skip-reason CSV (None -> output_dir/BreathCSV_Report.csv)
    """
    import os
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    os.makedirs(output_dir, exist_ok=True)
    if report_path is None:
        report_path = os.path.join(GS.Dir_Outputs(user_id), "BreathCSV_Report.csv")

    # ---- Setup: indices and ranges for safe mapping ----
    # Index by frames and get the corresponding iframe and src (output_00000.csv)
    # Get min/max of frame, and access frames_idx_iframe etc. only within that range
    frames_idx_iframe = frames.set_index("Frame")["iFrame"]
    frames_idx_src    = frames.set_index("Frame")["source_file"]
    # v1.4: Abnormal is set by initial_ocr (if the column is missing, treat as no abnormality)
    frames_idx_abn = (pd.to_numeric(frames["Abnormal"], errors="coerce").fillna(0).astype(int)
                      if "Abnormal" in frames else pd.Series(0, index=frames.index))
    frames_idx_abn.index = frames["Frame"]
    frame_min = int(frames["Frame"].min())
    frame_max = int(frames["Frame"].max())

    skip_logs = []
    summary_records = []

    # Axis scale: accumulate per breath {BreathNumber: {name: {min,max,unit}}}.
    # Ticks are determined together with neighboring breaths (_estimate_scales), so first collect all breaths' pixel waveforms, then convert and write.
    # reader is the argument formerly used for OCR of axis numbers (kept for compatibility; unused)
    scale_enabled = (settings or {}).get("post_ocr", {}).get("scale", {}).get("enabled", True)
    breath_scales = {}
    records = []        # in breath order: {breath, waves (pixels), zero, ppeak, vti, edi_peak, edi_full, df, out_path, ...}
    sweep_rates = []    # sweep speed (px/s) derived from sweep positions and times in start/end images
    # Diagnostics: per breath, store left/right x of the waveform region used for extraction and the sweep cursor x of start/end frames
    # {BreathNumber: (x_start, x_end, cursor_start, cursor_end)}. For verifying Sweep detection in Label.
    breath_xspan = {}
    # Last successful Edi scale (reused for breaths that cannot be calibrated because EdiPeak is *** = NaN)
    last_edi_scale = None


    # Start side: clip Frame-6 before mapping (avoid going negative at the beginning)
    
    # Convert the Breaths Frame column to numeric just in case, as a Series base_frame. I.e. base_frame is each breath's start frame number
    base_frame = pd.to_numeric(breaths["Frame"], errors="coerce")

    # If base_frame contains NaN etc., replace with frame_min; astype converts floats to int (17.0->17); clip replaces out-of-range (e.g. base_frame 3 minus 6 = -3) with frame_min
    start_back = max(1, round(START_BACK_SEC * GS.FPS))   # v1.4: at 10 fps, the same 6 frames as v1.3
    adjusted_frame_start = (base_frame - start_back).fillna(frame_min).astype(int).clip(frame_min, frame_max)
    breaths["_Start_Frame"] = adjusted_frame_start   # v1.3: for computing waveform time (Wave_T0). Removed at the end


    # Identify the start frame
    # Start_File looks up frames_idx_iframe keyed by the value stored in adjusted_frame_start, so start_file gets the start iframe
    # Start_Folder uses the same logic and ends up with a subfolder name like [00000]
    breaths["Start_File"] = adjusted_frame_start.map(frames_idx_iframe)
    breaths["Start_Folder"] = (
        adjusted_frame_start.map(frames_idx_src)
        .str.replace("output_", "", regex=False)
        .str.replace(".csv",   "", regex=False)
    )

    # Identify the end frame
    # End side: use the next breath's Frame as is (last row is removed since it has no End)
    breaths["End_Frame"] = breaths["Frame"].shift(-1)
    breaths = breaths.iloc[:-1]   # 0.1.1: so it does not fail on drop(index[-1]) when there are no breaths

    adjusted_frame_end = pd.to_numeric(breaths["End_Frame"], errors="coerce").fillna(frame_max).astype(int)
    adjusted_frame_end = adjusted_frame_end.clip(frame_min, frame_max)

    breaths["_End_Frame"] = adjusted_frame_end   # v1.4: to check whether the end image is Abnormal. Removed at the end
    breaths["End_File"] = adjusted_frame_end.map(frames_idx_iframe)
    breaths["End_Folder"] = (
        adjusted_frame_end.map(frames_idx_src)
        .str.replace("output_", "", regex=False)
        .str.replace(".csv",   "", regex=False)
    )
    breaths.drop(columns=["End_Frame"], inplace=True)

    # Safely drop mapping failures (NaN)
    bad = breaths[breaths[["Start_File", "End_File"]].isna().any(axis=1)]
    if not bad.empty:
        for _, r in bad.iterrows():
            skip_logs.append({
                "BreathNumber": int(r.get("BreathNumber", -1)),
                "Status": "SKIPPED",
                "Reason": "boundary map NaN",
                "Details": f"Start={r.get('Start_File')}, End={r.get('End_File')}",
            })
        breaths = breaths.drop(bad.index)

    # Build image paths (e.g. .\users\admin\Frames/00000/frame_00000039.jpg)
    breaths["Start_File_Name"] = (
        frames_root + "/" + breaths["Start_Folder"] + "/frame_" +
        breaths["Start_File"].astype(int).astype(str).str.zfill(8) + ".jpg"
    )
    breaths["End_File_Name"] = (
        frames_root + "/" + breaths["End_Folder"] + "/frame_" +
        breaths["End_File"].astype(int).astype(str).str.zfill(8) + ".jpg"
    )

    # Exclude the first breath
    breaths = breaths[breaths["BreathNumber"] != 1].reset_index(drop=True)

    # Rows may have been dropped due to mapping failures, so reset the index before iterrows
    breaths = breaths.reset_index(drop=True)

    # v1.5: open the images to be used (start, end, and the one before each) up front, in ascending order of segment and frame number.
    # The app (engine_api 3) builds opened images from segment videos, and forward reads can proceed without re-seeking.
    # The processing below opens end -> start per breath (going backward), so v1.4 caused a segment-video seek every time
    wanted = set()
    for _, r in breaths.iterrows():
        for folder_col, file_col in (("Start_Folder", "Start_File"), ("End_Folder", "End_File")):
            try:
                folder, iframe = str(r[folder_col]), int(r[file_col])
            except (TypeError, ValueError):
                continue
            wanted.update((folder, i) for i in (iframe - 1, iframe) if i >= 0)
    for folder, iframe in sorted(wanted):
        os.path.isfile(f"{frames_root}/{folder}/frame_{iframe:08d}.jpg")

    # ---- Process each breath ----
    _bt_total = len(breaths)
    for _bt_i, (_, row) in enumerate(tqdm(breaths.iterrows(), total=_bt_total), 1):
        if _bt_i % 20 == 0 or _bt_i == _bt_total:
            _report_progress(user_id, analyze_id,
                             message=f"Extracting breath CSVs ({_bt_i}/{_bt_total})",
                             current=_bt_i, total=_bt_total)
        # Get and validate boundary values
        try:
            breath_num   = int(row["BreathNumber"])
            start_iframe = int(row["Start_File"])
            end_iframe   = int(row["End_File"])
        except Exception as e:
            skip_logs.append({
                "BreathNumber": int(row.get("BreathNumber", -1)),
                "Status": "SKIPPED",
                "Reason": "missing boundary columns",
                "Details": str(e),
            })
            continue


        # Filter Frames by BreathNumber to extract only this breath's data

        filtered = frames[frames["BreathNumber"] == breath_num].copy()

        # Error if empty
        if filtered.empty:
            skip_logs.append({
                "BreathNumber": breath_num, "Status": "SKIPPED",
                "Reason": "empty frame range", "Details": f"{start_iframe}-{end_iframe}",
            })
            continue

        # v1.4: if the start/end images used to read the waveform are Abnormal (e.g. popups), do not write.
        # Abnormal frames mid-breath don't matter (fine as long as the waveform can be read from the end image; same idea as Hamilton).
        # Up to v1.3, a breath was not written if even one frame within it was Abnormal
        abn_start = int(frames_idx_abn.get(int(row["_Start_Frame"]), 0))
        abn_end = int(frames_idx_abn.get(int(row["_End_Frame"]), 0))
        if abn_start or abn_end:
            skip_logs.append({
                "BreathNumber": breath_num, "Status": "SKIPPED",
                "Reason": "abnormal frames", "Details": f"start={abn_start}, end={abn_end}",
            })
            continue

        # Check image existence
        img_start = str(row["Start_File_Name"])
        img_end   = str(row["End_File_Name"])
        if not (os.path.isfile(img_start) and os.path.isfile(img_end)):
            skip_logs.append({
                "BreathNumber": breath_num, "Status": "SKIPPED",
                "Reason": "image not found",
                "Details": f"start={os.path.isfile(img_start)}, end={os.path.isfile(img_end)}",
            })
            continue

        # Extract difference regions
        regions = get_difference_regions(img_end, img_start)
        if not regions:
            skip_logs.append({
                "BreathNumber": breath_num, "Status": "SKIPPED",
                "Reason": "no difference regions", "Details": "",
            })
            continue

        # Trim previous-sweep afterimage right of the end cursor (tip detected by diff with the previous frame).
        # For a true wrap (separate left/right regions), do not trim; defer to _select_breath_regions.
        # The previous frame is derived as (iframe-1) in the same folder as img_end.
        cursor_x = None
        before_path = os.path.join(os.path.dirname(img_end),
                                   f"frame_{end_iframe - 1:08d}.jpg")
        if end_iframe - 1 >= 0 and os.path.isfile(before_path):
            cursor_x = _detect_sweep_cursor(img_end, before_path)
            regions = _trim_regions_right_of_cursor(regions, cursor_x)

        regions = _select_breath_regions(regions)   # remove afterimage fragments on sweep wrap

        # Sweep cursor x on the start-frame side (detected by diff between start frame and previous frame)
        cursor_x_start = None
        before_start = os.path.join(os.path.dirname(img_start),
                                    f"frame_{start_iframe - 1:08d}.jpg")
        if start_iframe - 1 >= 0 and os.path.isfile(before_start):
            cursor_x_start = _detect_sweep_cursor(img_start, before_start)

        # Diagnostics: record left/right x of the extraction region and detected cursor x of start/end frames (output to Breaths.csv)
        if regions:
            breath_xspan[breath_num] = (
                int(min(r[0] for r in regions)),
                int(max(r[1] for r in regions)),
                int(cursor_x_start) if cursor_x_start is not None else None,
                int(cursor_x) if cursor_x is not None else None,
            )

        # Waveform extraction
        waveforms = merge_waveforms_across_regions(img_end, regions)
        waveforms = _trim_trailing_breath(waveforms) # trim the next breath's rise at the end
        waveforms = _reject_volume_troughs(waveforms) # remove non-physiological V-troughs in Volume
        length = len(next(iter(waveforms.values()), []))
        if length == 0:
            skip_logs.append({
                "BreathNumber": breath_num, "Status": "SKIPPED",
                "Reason": "empty waveform", "Details": "",
            })
            continue

        # Sweep speed (used to estimate Flow ticks). Breaths containing a wrap or spanning segments are not used
        if cursor_x_start is not None and cursor_x is not None and cursor_x > cursor_x_start \
                and row["Start_Folder"] == row["End_Folder"] and end_iframe > start_iframe:
            sweep_rates.append((cursor_x - cursor_x_start) / ((end_iframe - start_iframe) / GS.FPS))

        # Overwrite handling
        out_path = os.path.join(output_dir, f"{breath_num}.csv")
        if overwrite and os.path.exists(out_path):
            try:
                os.remove(out_path)
            except Exception as e:
                skip_logs.append({
                    "BreathNumber": breath_num, "Status": "SKIPPED",
                    "Reason": "overwrite failed", "Details": str(e),
                })
                continue

        # CSV output (drop last 30 points)
        df_out = pd.DataFrame({"X": list(range(1, length + 1))})
        for name in ["Pressure", "Flow", "Volume", "Edi"]:
            values = waveforms.get(name, [np.nan] * length)
            df_out[name] = values

        if len(df_out) <= 30:
            skip_logs.append({
                "BreathNumber": breath_num, "Status": "SKIPPED",
                "Reason": "too short after trim", "Details": f"len={len(df_out)}",
            })
            continue

        df_out = df_out.iloc[:-30].copy()
        img_end_bgr = _load_frame(img_end) if scale_enabled else None
        records.append({
            "breath": breath_num, "df": df_out, "out_path": out_path,
            "waves": {nm: df_out[nm].to_numpy(float) for nm in ["Pressure", "Flow", "Volume", "Edi"]},
            "zero": {nm: _zero_px(img_end_bgr, nm) for nm in ZERO_Y},
            "ppeak": row.get("Real_Ppeak"), "vti": row.get("Real_VTi"),
            "edi_peak": row.get("EdiPeak"), "edi_full": waveforms.get("Edi"),
            "start_frame": int(row["_Start_Frame"]),
            "x0": int(min(r[0] for r in regions)),
            "cursor_start": int(cursor_x_start) if cursor_x_start is not None else None,
            "summary": {
                "BreathNumber": breath_num,
                "Start_Frame_Image": img_start,
                "End_Frame_Image": img_end,
                "Regions": "; ".join([f"{x0}–{x1}" for (x0, x1) in regions]),
            },
        })

    # ---- Estimate axis scales, convert to physical units, and write per-breath CSVs ----
    rate = float(np.median(sweep_rates)) if len(sweep_rates) >= 3 else SWEEP_RATE_DEFAULT
    estimated = _estimate_scales(records, rate) if scale_enabled else {}
    _trim_records_to_breaths(records, rate)   # ticks are estimated from waveforms before trimming
    breath_wave_time = {}
    for rec in records:
        breath_num = rec["breath"]
        df_out = rec["df"].iloc[rec["head"]:rec["tail"]].copy()   # X keeps the original column indices
        breath_wave_time[breath_num] = (rec["wave_t0"], rate, rec["head"], len(rec["df"]) - rec["tail"])
        if scale_enabled:
            scale = dict(estimated.get(breath_num, {}))
            # Edi is calibrated with the EdiPeak value
            edi_scale = _edi_scale_from_peak(rec["edi_peak"], rec["edi_full"])
            if edi_scale:
                scale["Edi"] = edi_scale
                last_edi_scale = edi_scale         # remember the valid scale
            elif last_edi_scale is not None:
                # Cannot calibrate because EdiPeak is *** (NaN) etc. -> reuse the Edi scale of the last valid breath
                scale["Edi"] = last_edi_scale
            for nm, sc in scale.items():
                df_out[nm] = _pixels_to_physical(df_out[nm].tolist(), nm, sc)
            if scale:
                breath_scales[breath_num] = scale
        df_out.to_csv(rec["out_path"], index=False)
        print(f"{rec['out_path']} created")

        # Summary/report
        summary_records.append(rec["summary"])
        skip_logs.append({
            "BreathNumber": breath_num, "Status": "OK", "Reason": "", "Details": ""
        })

    # ---- Expand per-breath scales into columns for Breaths.csv (P_min,P_max,F_min,...) ----
    if breath_scales:
        short = {"Pressure": "P", "Flow": "F", "Volume": "V", "Edi": "E"}
        def _sc(b, nm, key):
            try:
                return breath_scales.get(int(b), {}).get(nm, {}).get(key)
            except (ValueError, TypeError):
                return None
        for nm, pfx in short.items():
            breaths[f"{pfx}_min"] = breaths["BreathNumber"].map(lambda b, nm=nm: _sc(b, nm, "min"))
            breaths[f"{pfx}_max"] = breaths["BreathNumber"].map(lambda b, nm=nm: _sc(b, nm, "max"))

    # ---- Diagnostics: expand left/right x of extracted waveform region (X_Start/X_End) and detected cursor positions into columns ----
    def _xs(b, i):
        try:
            v = breath_xspan.get(int(b))
            return v[i] if v is not None else None
        except (ValueError, TypeError):
            return None
    breaths["X_Start"] = breaths["BreathNumber"].map(lambda b: _xs(b, 0))
    breaths["X_End"] = breaths["BreathNumber"].map(lambda b: _xs(b, 1))
    breaths["Sweep_Cursor_Start"] = breaths["BreathNumber"].map(lambda b: _xs(b, 2))
    breaths["Sweep_Cursor_End"] = breaths["BreathNumber"].map(lambda b: _xs(b, 3))

    # ---- v1.3: time of first waveform point and sweep speed (the app uses these to compute each point's time), rows trimmed ----
    def _wt(b, i):
        try:
            v = breath_wave_time.get(int(b))
            return v[i] if v is not None else None
        except (ValueError, TypeError):
            return None
    breaths["Wave_T0"] = breaths["BreathNumber"].map(lambda b: _wt(b, 0))
    breaths["Wave_Rate"] = breaths["BreathNumber"].map(lambda b: _wt(b, 1))
    breaths["Wave_Trim_Head"] = breaths["BreathNumber"].map(lambda b: _wt(b, 2))
    breaths["Wave_Trim_Tail"] = breaths["BreathNumber"].map(lambda b: _wt(b, 3))
    breaths = breaths.drop(columns=["_Start_Frame", "_End_Frame"])

    # ---- Report output ----
    # 0.1.1: when there were no candidate breaths to write (2 or fewer breath boundaries, e.g. mismatched video), skip_logs was empty and
    # sorting a column-less table raised KeyError and stopped the analysis. Make it a table with columns even when empty
    report_df = pd.DataFrame(skip_logs, columns=["BreathNumber", "Status", "Reason", "Details"]).sort_values("BreathNumber")
    report_df.to_csv(report_path, index=False)

    return len(summary_records), breaths


def re_extract_breath_waveform(img_start_path, img_end_path, end_iframe, scale=None):
    """Re-extract one breath's waveform from the diff of start/end frame images and return it as a DataFrame.

    Same procedure as the BreathsToCSVs waveform extraction core (diff regions -> trim afterimage right of end cursor -> sweep-wrap region selection ->
    region concatenation -> trim next breath at end -> Volume trough removal -> drop last 30 points).
    Used after breath merging (Fragments etc. merged with neighbors) to rebuild one clean breath waveform from the merged interval.
    Concatenating existing {n}.csv files stacks two breaths and breaks the waveform, so always re-extract with this function.
    v1.3: trimming to the breath span (_trim_records_to_breaths) needs timing info, so it is not done here.

    Parameters
    ----------
    img_start_path : str   start frame image path of the merged breath (= original Start_File_Name)
    img_end_path   : str   end frame image path of the merged breath (= End_File_Name, the next breath's Frame)
    end_iframe     : int   iframe number of the end frame (used to derive the previous frame for end-cursor detection)
    scale          : dict|None  {name: {"min":, "max":}} if given, convert pixels to physical units.

    Returns
    -------
    pandas.DataFrame | None   columns=[X, Pressure, Flow, Volume, Edi]. None if re-extraction is impossible.
    """
    if not (os.path.isfile(img_start_path) and os.path.isfile(img_end_path)):
        return None

    # Extract difference regions
    regions = get_difference_regions(img_end_path, img_start_path)
    if not regions:
        return None

    # Trim previous-sweep afterimage right of the end cursor (tip detected by diff with the previous frame)
    before_path = os.path.join(os.path.dirname(img_end_path),
                               f"frame_{int(end_iframe) - 1:08d}.jpg")
    if int(end_iframe) - 1 >= 0 and os.path.isfile(before_path):
        cursor_x = _detect_sweep_cursor(img_end_path, before_path)
        regions = _trim_regions_right_of_cursor(regions, cursor_x)

    regions = _select_breath_regions(regions)   # remove afterimage fragments on sweep wrap

    # Waveform extraction
    waveforms = merge_waveforms_across_regions(img_end_path, regions)
    waveforms = _trim_trailing_breath(waveforms)   # trim the next breath's rise at the end
    waveforms = _reject_volume_troughs(waveforms)  # remove non-physiological V-troughs in Volume
    length = len(next(iter(waveforms.values()), []))
    if length == 0:
        return None

    # Convert to physical units (scale={name:{min,max}}; reusing Breaths.csv P_min/P_max etc. avoids OCR)
    if scale:
        for nm in ["Pressure", "Flow", "Volume", "Edi"]:
            if nm in scale and nm in waveforms and scale[nm] is not None:
                waveforms[nm] = _pixels_to_physical(waveforms[nm], nm, scale[nm])

    df_out = pd.DataFrame({"X": list(range(1, length + 1))})
    for name in ["Pressure", "Flow", "Volume", "Edi"]:
        df_out[name] = waveforms.get(name, [np.nan] * length)

    if len(df_out) <= 30:   # too short if nothing remains after dropping the last 30 points (same criterion as BreathsToCSVs)
        return None
    return df_out.iloc[:-30].reset_index(drop=True)


def WaveFromCSV(csv_path):
    df = pd.read_csv(csv_path)
    
    # Sort by X column if present
    if "X" in df.columns:
        df = df.sort_values(by="X").reset_index(drop=True)
        globals()["x_start"] = df["X"].iloc[0]  # record in a global variable
    
    waveforms = {}
    for name in ["Pressure", "Flow", "Volume", "Edi"]:
        if name in df.columns:
            waveforms[name] = df[name].to_numpy()
        else:
            print(f"Warning: {name} column not found. Filling with NaN.")
            waveforms[name] = np.full(len(df), np.nan)
   


    return waveforms

def PlotWave(waveforms, x_vals=None, reference_waveforms=None):
    wave_names = [k for k in ["Pressure", "Flow", "Volume", "Edi"] if k in waveforms and not k.lower().startswith("original")]
    num_points = len(next(iter(waveforms.values())))
    if x_vals is None:
        x_vals = np.arange(1, num_points + 1)
    else:
        x_vals = np.asarray(x_vals)

    rows = 2 if reference_waveforms else 1
    fig, axes = plt.subplots(rows * len(wave_names), 1, figsize=(12, 2.5 * rows * len(wave_names)), sharex=True)
    axes = np.array(axes).reshape((rows, len(wave_names))) if rows > 1 else np.array([axes])

    for i, name in enumerate(wave_names):
        for row, data in enumerate(([waveforms] + ([reference_waveforms] if reference_waveforms else []))):
            ax = axes[row][i] if rows > 1 else axes[0][i]
            y_vals = np.asarray(data.get(name, np.full(len(x_vals), np.nan)))
            isnan = np.isnan(y_vals)
            idx = np.where(~isnan)[0]
            segments = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
            for seg in segments:
                ax.plot(x_vals[seg], y_vals[seg], label=name)
            if len(idx) > 0:
                first_x = x_vals[idx[0]]
                ax.axvline(first_x, color="red", linestyle="--", alpha=0.3)
                ax.text(first_x + 2, np.nanmax(y_vals) * 0.95, f"X={first_x}", color="red", fontsize=8)
            ax.set_ylabel("Height (px)")
            ax.set_title(f"{name} {'(Reference)' if row == 1 else ''}")
            ax.grid(True)

    axes[-1][0].set_xlabel("X-coordinate (pixel)")
    tick_indices = np.linspace(0, len(x_vals) - 1, num=10, dtype=int)
    axes[-1][0].set_xticks(x_vals[tick_indices])
    axes[-1][0].set_xticklabels([f"{x_vals[i]}" for i in tick_indices])

    plt.tight_layout()
    plt.show()


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
