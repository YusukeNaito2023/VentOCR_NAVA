from tqdm import tqdm
from easyocr import Reader
from . import Settings
import cv2 #pip install opencv-python-headless (not from anaconda!)
import re
import pandas as pd
import gc
import os
import numpy as np
import glob
import Global_Setting as GS
from . import ExtractWaves as ew
from . import sweep
from . import glyphs
from pathlib import Path
import threading
import time

ENGINE_VERSION = "0.2.0"
VENTOCR_TESTED = "0.5.5"

_GRAY_WEIGHTS = [0.114, 0.587, 0.299]   # Same formula as the full-frame grayscale conversion up to v1.4 (np.dot -> truncated to uint8)


def _resize_period(src_wh, base_wh):
    """Downscale period (in pixels at the base resolution). 1920->1280 is 3:2, so 2. None if not divisible within 1-16."""
    for p in range(1, 17):
        if all((p * s) % b == 0 for s, b in zip(src_wh, base_wh)):
            return p
    return None


class _BaseFrame:
    """View a frame at the base resolution (Settings.BASE_RESOLUTION). frame[y1:y2, x1:x2] returns only that region.

    v1.5: Up to v1.4 the whole frame was downscaled to the base resolution (about 8 ms per frame at 1920x1080; about 40% of segment processing time).
    If the scale ratio is rational, downscaling a period-aligned region with INTER_AREA yields the same pixels as downscaling the whole frame.
    When the region cannot be aligned to the period (including upscaling), the whole frame is downscaled once, as in v1.4."""

    def __init__(self, src):
        self.src = src
        base_w, base_h = Settings.BASE_RESOLUTION
        h, w = src.shape[:2]
        self.full = src if (w, h) == (base_w, base_h) else None
        self.period = _resize_period((w, h), (base_w, base_h)) if w > base_w else None
        if self.full is None and self.period is None:
            interp = cv2.INTER_AREA if w > base_w else cv2.INTER_LINEAR
            self.full = cv2.resize(src, (base_w, base_h), interpolation=interp)
        self.sx, self.sy = w / base_w, h / base_h

    def mean(self):
        """For black-screen detection (downscaling barely changes the mean, so compute it on the original image)."""
        return float((self.full if self.full is not None else self.src).mean())

    def __getitem__(self, sl):
        if self.full is not None:
            return self.full[sl]
        ys, xs = sl
        y, y2, x, x2, p = ys.start, ys.stop, xs.start, xs.stop, self.period
        y0, x0 = y - y % p, x - x % p
        y1, x1 = -(-y2 // p) * p, -(-x2 // p) * p
        crop = self.src[round(y0 * self.sy):round(y1 * self.sy), round(x0 * self.sx):round(x1 * self.sx)]
        out = cv2.resize(crop, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
        return out[y - y0:y2 - y0, x - x0:x2 - x0]


class _GrayView:
    """Convert only image[y1:y2, x1:x2] to grayscale (up to v1.4 the whole frame was computed in float64; about 8 ms per frame)."""

    def __init__(self, frame):
        self.frame = frame

    def __getitem__(self, sl):
        return np.dot(self.frame[sl][..., :3], _GRAY_WEIGHTS).astype(np.uint8)


_EASYOCR_LOCKFILE = "easyocr_init.lock"


def _get_easyocr_model_dir():
    # Use a persistent storage location shared per user, not per Engine
    model_dir = os.path.join(".", "_runtime_models", "easyocr")
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def _easyocr_model_ready(model_dir: str) -> bool:
    p = Path(model_dir)
    if not p.exists():
        return False

    # Consider it minimally OK if at least one EasyOCR detection/recognition model is present
    patterns = [
        "craft*.pth",
        "dbnet*.pth",
        "latin*.pth",
        "english_g2*.pth",
        "standard*.pth"
    ]
    for pat in patterns:
        if list(p.glob(pat)):
            return True
    return False


def _acquire_easyocr_file_lock(model_dir: str, timeout: int = 300, poll: float = 0.5):
    lock_path = os.path.join(model_dir, _EASYOCR_LOCKFILE)
    start = time.time()

    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            return fd, lock_path
        except FileExistsError:
            if (time.time() - start) > timeout:
                raise TimeoutError(f"Timeout waiting for EasyOCR lock: {lock_path}")
            time.sleep(poll)


def _release_easyocr_file_lock(fd, lock_path: str):
    try:
        os.close(fd)
    except Exception:
        pass
    try:
        if os.path.exists(lock_path):
            os.remove(lock_path)
    except Exception:
        pass


_READER_CACHE = {}  # v1.5: (gpu, model folder) -> Reader


def _ensure_easyocr_reader_ready(gpu_enabled: bool):
    """v1.5: Reuse the Reader within the same process (the app's parallel workers).
    Up to v1.4 the models were reloaded for every segment (about 1.8 s each; 60 times for a 1-hour video)."""
    key = (bool(gpu_enabled), _get_easyocr_model_dir())
    if key not in _READER_CACHE:
        _READER_CACHE[key] = _create_easyocr_reader(gpu_enabled)
    return _READER_CACHE[key]


def _create_easyocr_reader(gpu_enabled: bool):
    model_dir = _get_easyocr_model_dir()

    # If already complete, use it immediately
    if _easyocr_model_ready(model_dir):
        print(f"[EasyOCR] using existing model: {model_dir}")
        return Reader(
            ["en"],
            gpu=gpu_enabled,
            model_storage_directory=model_dir,
            download_enabled=True
        )

    # If not ready, take an inter-process lock
    fd, lock_path = _acquire_easyocr_file_lock(model_dir)
    try:
        # Another process may have finished while we were waiting, so check again
        if not _easyocr_model_ready(model_dir):
            print(f"[EasyOCR] model not found. downloading to: {model_dir}")
            Reader(
                ["en"],
                gpu=gpu_enabled,
                model_storage_directory=model_dir,
                download_enabled=True
            )
        else:
            print(f"[EasyOCR] model became ready while waiting: {model_dir}")
    finally:
        _release_easyocr_file_lock(fd, lock_path)

    return Reader(
        ["en"],
        gpu=gpu_enabled,
        model_storage_directory=model_dir,
        download_enabled=True
    )

_glyph_reader = None


def _get_glyph_reader():
    """Template reader of the numeric ROIs (one per process); None without the template file."""
    global _glyph_reader
    if _glyph_reader is None:
        _glyph_reader = glyphs.load_reader(os.path.dirname(os.path.abspath(__file__)), Settings.VALUE_GLYPH_FILE) or False
    return _glyph_reader or None


def _roi_changed(a, b):
    """The displayed value changed: at least 0.5% of the pixels changed by 24 gray levels or more (video noise never does)."""
    return (np.abs(a.astype(np.int16) - b.astype(np.int16)) >= 24).mean() >= 0.005


def _keep_frame(keep_dir, i_frame, base_frame):
    """Write the waveform area (ew.KEEP_BOX, BASE_RESOLUTION) of a frame for post_ocr (ew._load_frame reads it back)."""
    x1, y1, x2, y2 = ew.KEEP_BOX
    cv2.imwrite(os.path.join(keep_dir, f"frame_{i_frame:08d}.jpg"), base_frame[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 95])


def initial_ocr(frames, total_frames, subfolder, settings={}, progress_callback=None, frames_dir=None):
    """
    v1.1: Supports a color frame generator.
    - Removed gc.collect() from inside the loop
    - Batch OCR of ROIs
    - Receives color frames from a generator and converts them to grayscale internally

    Args:
        frames: Generator of color frames (BGR, numpy arrays). Do not convert to a list.
        frames_dir: str, the frame image folder of the segment (0.2.0: its parent work folder gets ew.KEEP_DIR_NAME)
    """
    from easyocr import Reader

    skip_keys = settings["initial_ocr"].get("skip_keys", [])
    gpu_enabled = settings["initial_ocr"].get("use_gpu", True)

    # Diff-gating settings (with defaults so that old engine_options without them still work)
    _io = settings["initial_ocr"]
    diff_gating = _io.get("diff_gating", True)
    diff_thresh = _io.get("diff_thresh", 2.0)
    diff_thresh_cdyn = _io.get("diff_thresh_cdyn", 0.5)  # Low threshold for Cdyn, to stay on the safe side
    # If recognize's conf is below this threshold, fall back to readtext (with detection)
    recognize_conf_fallback = _io.get("recognize_conf_fallback", 0.5)
    # 0.2.0: the numeric ROIs are read by digit template matching on every frame (glyphs.py, value_glyphs.npz);
    # EasyOCR is used only when the templates cannot read an ROI (and then only when the ROI changed: diff gating)
    glyph_reader = _get_glyph_reader() if _io.get("glyph_read", True) else None
    # 0.2.0: images for post_ocr are kept here while the frames pass (see _keep_frame), so post_ocr does not have the
    # app decode the video again
    keep_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(frames_dir))), ew.KEEP_DIR_NAME, subfolder) \
        if frames_dir and _io.get("keep_images", True) else None
    if keep_dir:
        os.makedirs(keep_dir, exist_ok=True)

    reader = _ensure_easyocr_reader_ready(gpu_enabled)
    print(f"GPU:{gpu_enabled}")

    # ---- Batch OCR configuration ----
    OCR_BATCH_SIZE = 16

    # Targets for skipping detection (recognize): single-number ROIs.
    # IE_Ratio keeps readtext because it has multiple tokens ("1 : 2.3") / detection is useful.
    RECOGNIZE_KEYS = {"VTe", "EdiPeak", "Cdyn", "Ppeak", "VTi"}
    # 0.2.0: every ROI is a numeric display that changes only per breath, so all of them are checked on every frame
    # and OCRed only when the ROI image changed (diff gating). The Triggered ROI (patient-trigger icon) is no longer
    # OCRed: up to 0.1.x it was read with readtext every 10 frames, which never returned text for an icon.
    # Patient triggers are now detected from the white parts of the waveforms (ExtractWaves).
    GATED_KEYS = {"VTe", "EdiPeak", "Cdyn", "IE_Ratio", "Ppeak", "VTi"}

    Data = []
    frame_count = 0

    # Pre-compute ROI keys and positions once
    roi_items = list(Settings.ROIs.items())
    all_keys = [k for k, _ in roi_items]
    extra_keys = [f"{k}_raw" for k in all_keys] + [f"{k}_conf" for k in all_keys]
    # 0.2.0: pen tip of every frame (x in pixels of the received frame, NaN if not detected) and the frame width,
    # used by ExtractWaves to rebuild each breath from its last image (same approach as the Hamilton engine)
    # OCR_Count: number of ROIs actually OCRed on the frame (the others reused the last result; diff gating)
    column_order = all_keys + extra_keys + ["Sweep_Tip_Raw", "Frame_Width", "OCR_Count"]

    # ---- Batch accumulator ----
    # Each entry: (frame_idx, key, roi_image)
    batch_rois = []
    batch_meta = []  # (frame_idx_in_pending, key, preprocess_type)
    # pending_frames[i] = extracted_values dict for frame i
    pending_frames = []
    pending_start_frame = 0

    # State for diff gating (per ROI key)
    prev_roi = {}        # key -> ROI (ndarray) at the last OCR
    last_ocr_idx = {}    # key -> pending_frames index where that ROI was OCRed
    reuse_links = []     # (target_frame_rel, key, source_frame_rel)

    # v1.4: Abnormal frames (pop-ups, etc.) are detected here and become a column in Frames.csv.
    # Up to v1.3, DetectAbnormalFrames in post_ocr re-read the JPEGs of all frames
    abnormal_detector = ew.AbnormalFrameDetector()
    prev_trigger_rois, prev_base_frame, keep_pending = None, None, set()
    sweep_detector = sweep.SweepDetector()

    def _flush_batch():
        """Send accumulated ROIs to EasyOCR in one batch call, then dispatch results."""
        if not batch_rois:
            return

        # readtext_batched: call readtext once per ROI but reuse GPU session
        # EasyOCR doesn't support true multi-image batch, but we minimize
        # Python↔GPU round-trips by pre-loading all ROIs into a list and
        # calling readtext in tight succession without gc/callback interrupts.
        results = []
        for roi_img, (frame_rel, key, pp_type) in zip(batch_rois, batch_meta):
            if key in RECOGNIZE_KEYS:
                # Skip detection (CRAFT) and run recognition (CRNN) only. Read the whole ROI as one line
                rc = reader.recognize(roi_img)
                conf_rc = max((c for _, _, c in rc), default=0.0)
                # On low confidence, fall back to readtext (with detection) and adopt the result with higher confidence.
                # Because recognize occasionally mistakes edge noise in tight numeric ROIs for a digit (conf is low then).
                if conf_rc < recognize_conf_fallback:
                    rd = reader.readtext(roi_img)
                    conf_rd = max((c for _, _, c in rd), default=0.0)
                    results.append(rd if conf_rd >= conf_rc else rc)
                else:
                    results.append(rc)
            else:
                results.append(reader.readtext(roi_img))

        for idx, (frame_rel, key, pp_type) in enumerate(batch_meta):
            result = results[idx]
            ev = pending_frames[frame_rel]

            if key == "IE_Ratio":
                raw_texts = [text for (_, text, _) in result] if result else []
                confidences = [score for (_, _, score) in result] if result else []
                ev["IE_Ratio"] = ""
                ev["IE_Ratio_raw"] = ", ".join(raw_texts)
                ev["IE_Ratio_conf"] = ", ".join(f"{s:.2f}" for s in confidences)
            else:
                value, raw_text, confidence = _parse_ocr_result(result)
                ev[key] = value if _is_clean_number(key, raw_text) else ""
                ev[f"{key}_raw"] = raw_text
                ev[f"{key}_conf"] = confidence

        batch_rois.clear()
        batch_meta.clear()

    def _finalize_pending():
        """Move pending_frames into Data and clear."""
        nonlocal pending_start_frame
        for ev in pending_frames:
            fnum = ev["_frame"]
            tmp_list = [fnum] + [ev.get(k) for k in column_order]
            Data.append(tmp_list)
        pending_frames.clear()
        pending_start_frame = frame_count

    with tqdm(total=total_frames, desc="Analyzing Frames") as pbar:

        if progress_callback is not None:
            progress_callback(current=0, total=total_frames, message="Running OCR")

        # Receive frames one by one from the color frame generator and convert to grayscale
        for color_frame in frames:
            tip_raw = sweep_detector.add(color_frame)   # in pixels of the received frame
            frame_width = color_frame.shape[1]
            # Resolution independence: normalize to the base resolution of the ROI definitions (Settings.BASE_RESOLUTION).
            # v1.5: Do not downscale the whole frame; downscale/grayscale only the regions used (_BaseFrame; results are identical to full-frame computation)
            color_frame = _BaseFrame(color_frame)
            image = _GrayView(color_frame)

            frame_rel = len(pending_frames)
            extracted_values = {"_frame": frame_count, "Sweep_Tip_Raw": tip_raw, "Frame_Width": frame_width, "OCR_Count": 0}
            pending_frames.append(extracted_values)
            abnormal_detector.add(color_frame)   # v1.4: abnormal-frame detection (same order as Data)

            trigger_rois = {}
            for key, (x, y, w, h) in roi_items:
                if key in skip_keys:
                    extracted_values[key] = None
                    extracted_values[f"{key}_raw"] = ""
                    extracted_values[f"{key}_conf"] = "0.00"
                    continue

                roi = image[y:y+h, x:x+w]
                if key in ew.KEEP_TRIGGER_KEYS:
                    trigger_rois[key] = roi

                if glyph_reader is not None:
                    g = glyph_reader.read(roi, key)
                    if g is None and key in Settings.WIDE_ROIS:
                        # 0.2.1: a 4-digit value is cut by the ROI; read the wider ROI (only when the ROI read failed)
                        wx, wy, ww, wh = Settings.WIDE_ROIS[key]
                        g = glyph_reader.read(image[wy:wy+wh, wx:wx+ww], key + glyphs.WIDE_SUFFIX)
                    if g is not None:
                        text, score = g
                        if key == "IE_Ratio":
                            extracted_values[key], extracted_values[f"{key}_raw"] = "", text
                        else:
                            extracted_values[key], extracted_values[f"{key}_raw"] = text, text
                        extracted_values[f"{key}_conf"] = f"{score:.2f}"
                        continue

                # Diff gating: if the ROI is nearly identical to the one at the last OCR, skip OCR and
                # copy the result of the last OCRed frame at finalize time.
                # 0.2.0: the comparison uses the ROI before the EdiPeak preprocessing, so the preprocessing runs only
                # for ROIs that are actually OCRed (up to 0.1.x it ran on every frame)
                if diff_gating and key in GATED_KEYS and key in prev_roi \
                        and prev_roi[key].shape == roi.shape:
                    thresh = diff_thresh_cdyn if key == "Cdyn" else diff_thresh
                    mad = np.abs(roi.astype(np.int16) - prev_roi[key].astype(np.int16)).mean()
                    if mad < thresh:
                        reuse_links.append((frame_rel, key, last_ocr_idx[key]))
                        continue

                if key in GATED_KEYS:
                    prev_roi[key] = roi.copy()
                    last_ocr_idx[key] = frame_rel
                if key == "EdiPeak":
                    roi = _preprocess_for_edi(roi)
                extracted_values["OCR_Count"] += 1
                batch_rois.append(roi)
                batch_meta.append((frame_rel, key, key))

            # Keep the images post_ocr will read: the last frame before a display update (the image of a breath) and
            # KEEP_AFTER frames after it (read again when the next breath's onset is not on the screen yet)
            if keep_dir:
                updated = prev_trigger_rois is not None and any(
                    k in prev_trigger_rois and _roi_changed(r, prev_trigger_rois[k]) for k, r in trigger_rois.items())
                if updated and prev_base_frame is not None:
                    _keep_frame(keep_dir, frame_count - 1, prev_base_frame)
                    keep_pending.update(frame_count + d for d in ew.KEEP_AFTER)
                if frame_count in keep_pending:
                    _keep_frame(keep_dir, frame_count, color_frame)
                    keep_pending.discard(frame_count)
                prev_trigger_rois, prev_base_frame = trigger_rois, color_frame

            # Flush batch when it reaches the threshold
            if len(batch_rois) >= OCR_BATCH_SIZE:
                _flush_batch()

            frame_count += 1
            pbar.update(1)

            # Progress callback every 50 frames (not every frame)
            if progress_callback is not None and (frame_count % 50 == 0 or frame_count == total_frames):
                progress_callback(current=frame_count, total=total_frames, message="Running OCR")

        # Flush remaining batch
        _flush_batch()
        if keep_dir and prev_base_frame is not None:
            _keep_frame(keep_dir, frame_count - 1, prev_base_frame)   # last frame of the segment (breaths across segments)

        # For frames reused by diff gating, apply the OCR result of the source frame.
        # pending_frames are kept until the end of the segment, so the source values are already final.
        for tgt, key, src in reuse_links:
            ev_t = pending_frames[tgt]
            ev_s = pending_frames[src]
            ev_t[key] = ev_s.get(key)
            ev_t[f"{key}_raw"] = ev_s.get(f"{key}_raw", "")
            ev_t[f"{key}_conf"] = ev_s.get(f"{key}_conf", "0.00")

        _finalize_pending()

    columns = ["Frame"] + column_order
    df = pd.DataFrame(Data, columns=columns)
    df[["Abnormal", "MeanDiff", "StdDiff", "HistCorr"]] = pd.DataFrame(
        abnormal_detector.results(), index=df.index,
        columns=["Abnormal", "MeanDiff", "StdDiff", "HistCorr"])
    return df

        
        
def post_ocr(df, Analyze_Id, user_ID, settings={}) :

    # Helper that forwards each sub-stage inside post_ocr to the overall progress (so it does not look stalled).
    # Reuses the existing ExtractWaves._report_progress (progress_io based, best effort).
    def _pp(msg, cur=0, tot=0):
        try:
            ew._report_progress(user_ID, Analyze_Id, message=msg, current=cur, total=tot)
        except Exception:
            pass

    _pp(f"post_ocr: assigning breath numbers ({len(df)} rows)")
    df = df.sort_values("Frame").reset_index(drop=True)
    df["IE_Ratio"] = df["IE_Ratio_raw"].apply(parse_IE_ratio)
    for key in ("VTe", "EdiPeak", "Cdyn", "Ppeak", "VTi"):
        if key in df.columns:
            df[key] = pd.to_numeric(df[key], errors="coerce")
    # 0.2.1: a VTe / VTi read above VOLUME_MAX_ML is not a value (a frame caught while the display changes, e.g. "11435"
    # between 0.0 and 1435); the frame is left empty, so the breath keeps its last good reading
    for key in ("VTe", "VTi"):
        if key in df.columns:
            df.loc[df[key] > VOLUME_MAX_ML, key] = np.nan

    # 0.2.0: breath boundaries are the ventilator's display updates, unchanged (no merging or splitting)
    df["BreathMark"] = breath_boundaries(df, GS.FPS)
    df["BreathNumber"] = df["BreathMark"].cumsum() + (0 if len(df) and df["BreathMark"].iloc[0] == 1 else 1)
    df["Alert"] = df["BreathNumber"].map(breath_alerts(df["BreathNumber"]))

    # 0.2.0: pen tip of every frame (sweep tracking). Sweep_X is in pixels of the analyzed video (ENGINE_API §7.2)
    width = float(pd.to_numeric(df.get("Frame_Width"), errors="coerce").median()) if "Frame_Width" in df else float(Settings.BASE_RESOLUTION[0])
    sx = width / Settings.BASE_RESOLUTION[0]
    fit = sweep.fit_sweep(pd.to_numeric(df["Sweep_Tip_Raw"], errors="coerce").to_numpy(float) / sx, GS.FPS) \
        if "Sweep_Tip_Raw" in df else None
    if fit is not None:
        df["Sweep_U"] = np.round(fit["u"], 2)
        df["Sweep_X"] = np.round(fit["x"] * sx, 1)
        df["Sweep_Speed"] = round(fit["speed"] * GS.FPS, 3)     # BASE_RESOLUTION px per second
        print(f"[post_ocr] sweep {fit['speed'] * GS.FPS:.2f} px/s, detected {fit['detected'].mean() * 100:.1f}% of frames")

        
    _pp("post_ocr: aggregating times and per-breath representative values")
    # Write the approximate time in the video into the DF based on FPS
    frame_interval = 1 / GS.FPS
    df["Time"] = pd.to_timedelta(df["Frame"] * frame_interval, unit="s")
    df["Time"] = df["Time"].apply(lambda x: f"{x.seconds//3600}:{(x.seconds%3600)//60:02}:{x.seconds%60:02}:{int(x.microseconds/10000):02}")

   
   # Get VTi, VTe of the next breath (VTi etc. change midway, so take the first value; it reflects the actual tidal volume)
    df['Real_VTe'] = Shift_Values(df, "VTe")
    # 0.2.0: Cdyn is shown with the same delay as VTe (updated at the start of the next breath), so the value at the start
    # of the next breath is this breath's Cdyn ("***" -> NaN)
    if "Cdyn" in df.columns:
        df["Real_Cdyn"] = Shift_Values(df.assign(Cdyn=pd.to_numeric(df["Cdyn"], errors="coerce")), "Cdyn")
    # Ppeak / VTi are also updated at the end of a breath (= start of the next breath), so the value at the start of the next breath is that breath's value.
    # Used for estimating waveform axis scales (ExtractWaves._estimate_scales)
    for key in ("Ppeak", "VTi"):
        if key in df.columns:
            df[f"Real_{key}"] = Shift_Values(df, key)

    # Also create a per-breath dataframe (v1.4: abnormal-detection columns would only hold the first frame's value, so they are not put in Breaths.csv)
    df_breath = df.loc[df.groupby('BreathNumber')['Frame'].idxmin()]
    df_breath = df_breath.drop(columns=["Abnormal", "MeanDiff", "StdDiff", "HistCorr", "Sweep_Tip_Raw", "Sweep_U", "Sweep_X",
                                        "Sweep_Speed", "Frame_Width", "OCR_Count", "BreathMark"], errors="ignore")

    # 0.2.1: Cdyn is what the display shows at the end of the breath ("***" -> NaN; unreadable frames carry the last
    # reading). The Cdyn display often changes 1-3 frames after VTe (70 of 1010 breaths in the Servo-U 1-h video), so
    # the first frame of a breath still showed the previous breath's Cdyn.
    if "Cdyn" in df.columns:
        cdyn_state = pd.to_numeric(_display_state(df, "Cdyn"), errors="coerce")
        df_breath["Cdyn"] = df_breath["BreathNumber"].map(cdyn_state.groupby(df["BreathNumber"]).agg(lambda s: s.iloc[-1]))


    # Fill in Edi Peak
    # For each BreathNumber, get the "last non-NaN EdiPeak that appeared"
    last_peak_per_breath = (
        df.groupby("BreathNumber")["EdiPeak"]
        .apply(lambda s: s.dropna().iloc[-1] if not s.dropna().empty else np.nan)
    )

    # Update all rows with the representative value
    df_breath["EdiPeak"] = df_breath["BreathNumber"].map(last_peak_per_breath)


    # Fill in VTe: the last reading of the breath, except a value seen only on the breath's last frame (0.2.1: there the
    # display sometimes already shows the next breath's VTe, one frame before the boundary; 4 breaths in the Servo-U
    # 1-h video, e.g. 517 instead of 786)
    last_peak_per_breath = df.groupby("BreathNumber")["VTe"].apply(_last_reading_settled)

    # Update all rows with the representative value
    df_breath["VTe"] = df_breath["BreathNumber"].map(last_peak_per_breath)

    # IE_Ratio (Ti/Ttot): last non-NaN value of the breath. The display sometimes updates 1-2 frames after Cdyn / VTe,
    # so up to 0.2.0 development the first frame of a breath often still showed the previous breath's value.
    if "IE_Ratio" in df.columns:
        last_ie_per_breath = (
            df.groupby("BreathNumber")["IE_Ratio"]
            .apply(lambda s: s.dropna().iloc[-1] if not s.dropna().empty else np.nan)
        )
        df_breath["IE_Ratio"] = df_breath["BreathNumber"].map(last_ie_per_breath)

    # 0.2.1: Alert Disconnect (VTe about 0 with a large VTi and Cdyn "***", for several breaths)
    disconnected = breath_disconnect(df, df_breath)
    if disconnected:
        df_breath["Alert"] = [",".join(dict.fromkeys([x for x in str(a).split(",") if x and x != "nan"] + ["Disconnect"]))
                              if bn in disconnected else a for bn, a in zip(df_breath["BreathNumber"], df_breath["Alert"])]
        df["Alert"] = df["BreathNumber"].map(dict(zip(df_breath["BreathNumber"], df_breath["Alert"])))
        print(f"[post_ocr] Disconnect: {len(disconnected)} breaths")






    ###########################################################################################
    # Per-breath waveform analysis
    base_dir = os.path.join(".", user_ID)

    # v1.4: The Abnormal column is set by initial_ocr (JPEGs of all frames are not re-read)
    df_frames = df
    if "Abnormal" not in df_frames:
        df_frames["Abnormal"] = 0
    ew.set_config(settings)

    df_breath.to_csv("breath.csv", index=False)
    df_frames.to_csv("frames.csv")
    df_breath.to_csv("breath.csv")

    # Exclude breaths marked Abnormal and write the rest to individual CSVs
    _pp(f"post_ocr: writing per-breath CSVs ({len(df_breath)} breaths)")
    print(f"[post_ocr] BreathsToCSVs starting ({len(df_breath)} breaths)...")
    # Axis scales are estimated from Real_Ppeak / Real_VTi and the waveforms (axis numbers are not OCRed, so no reader is needed)
    Effective_Breaths, df_breath = ew.BreathsToCSVs(df_breath,df_frames,GS.Dir_Frames(user_ID),GS.Dir_BreathCSVs(user_ID), user_ID, analyze_id=Analyze_Id, settings=settings)
    print(f"[post_ocr] BreathsToCSVs done ({Effective_Breaths} effective breaths)")

    

 

    # Create fields.txt (section format)
    # The destination uses Archive_Dir_For, same as the archive processing in Main.py.
    # Prevents a bug where, when a separate destination disk (VentOCR_Archive_*) is selected, fields.txt/scale_meta.json
    # are left behind on the default (Temporary) side and Label cannot draw graphs.
    archive_dir = GS.Archive_Dir_For(user_ID, Analyze_Id)
    os.makedirs(archive_dir, exist_ok=True)

    fields_path = os.path.join(archive_dir, "fields.txt")
    with open(fields_path, "w", encoding="utf-8") as f:
        f.write("[waves]\n")
        for w in ["Pressure", "Flow", "Volume", "Edi"]:
            f.write(f"{w}\n")
        f.write("\n")
        f.write("[fields]\n")
        for field in ["Cdyn", "VTe", "EdiPeak", "Real_VTe", "Real_Cdyn", "Real_Ppeak", "Real_VTi", "Trigger", "Trigger_Source", "Alert"]:
            f.write(f"{field}\n")

    # Save scale_meta.json (Engine-specific scale adjustment settings: step width / unit / symmetry).
    # Label (Flask/JS) reads it from the Archive and applies per-Engine settings such as the +/- button steps.
    try:
        import json as _json
        with open(os.path.join(archive_dir, "scale_meta.json"), "w", encoding="utf-8") as f:
            _json.dump(ew.scale_meta(), f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[post_ocr] failed to write scale_meta.json: {e}")

    return df_frames, df_breath




def _parse_ocr_result(result):
    """Parse EasyOCR readtext() result into (value, raw_text, confidence_str).
    v1.1: extracted from read_text() for use in batch pipeline.
    """
    raw_texts = []
    confidences = []

    for res in result:
        _, text, score = res
        text = text.replace("_", ".").replace("・", ".").replace("I", "1").replace("：",":")
        raw_texts.append(text)
        confidences.append(score)

    full_text = ", ".join(raw_texts)
    confidence_str = ", ".join([f"{s:.2f}" for s in confidences])

    numeric_parts = [re.sub(r"[^0-9.:]", "", t) for t in raw_texts]
    numeric_parts = [t for t in numeric_parts if re.match(r"^\d+\.?\d*$", t)]

    if len(numeric_parts) == 1:
        cleaned_value = numeric_parts[0]
    elif len(numeric_parts) >= 2:
        if numeric_parts[0].endswith('.'):
            cleaned_value = numeric_parts[0] + numeric_parts[1]
        else:
            cleaned_value = numeric_parts[0] + '.' + numeric_parts[1]
    else:
        cleaned_value = ""

    return cleaned_value, full_text, confidence_str


def _is_clean_number(key, raw_text):
    """0.2.1: an EasyOCR read is used only when its text is a plain number ("461", "7.3"; Cdyn may end with the star
    the Servo-U puts after some values). Other characters mean a misread digit ("46]" for 461, "110z" for 1067,
    "31." for 31.7); the value is then left empty, so the breath takes its value from the frames that were read."""
    t = raw_text.replace(" ", "")
    if key == "Cdyn":
        t = t.rstrip("*")
    return re.fullmatch(r"\d+(\.\d+)?", t) is not None


def read_text(reader, roi):
    """Legacy wrapper: calls readtext then parses. Kept for compatibility."""
    result = reader.readtext(roi)
    return _parse_ocr_result(result)


def normalize_ie_ratio(raw_texts: list[str], numeric_parts: list[str]) -> str:
    """
    Strictly normalize the IE notation from raw OCR text.
      - Output is always '1:X' or 'X:1'
      - X is at least 1, with 1 decimal place (rounded)
    Expected misreads:
      '1 :.21'→'1:2.1', '1.21'→'1:2.1', '1.2'→'1:2.0', '1.2.0'→'1:2.0',
      '01:02.1'→'1:2.1', 'T:17'→'1:1.7', 'E.17'→'1:1.7', '1, 2.2'→'1:2.2'
    """
    import re

    # 1) Text normalization
    s = " ".join(raw_texts)
    s = s.upper()
    s = (s.replace("：", ":")
           .replace(",", ":")
           .replace("I", "1").replace("L", "1").replace("T", "1"))
    s = re.sub(r"\bE[.:]?", "1:", s)      # e.g. 'E.17' -> '1:17'
    s = re.sub(r"\s+", "", s)

    # 2) Helper to convert a one-side token to a number
    def side_to_float(tok: str) -> float | None:
        tok = re.sub(r"[^0-9.]", "", tok)
        if tok == "":
            return None
        # Two or more dots (e.g. '2.0.0') -> first digit + the rest as the fractional part
        if tok.count(".") >= 2:
            digits = tok.replace(".", "")
            if len(digits) == 1:
                return float(f"{digits}.0")
            return float(f"{digits[0]}.{digits[1:]}")
        # Leading dot (e.g. '.21') -> treat as a lost digit left of the decimal point -> '2.1'
        if tok.startswith("."):
            d = tok[1:]
            if d == "":
                return None
            if len(d) == 1:
                return float(f"{d}.0")
            return float(f"{d[0]}.{d[1:]}")
        # No dot, 2+ digits (e.g. '17') -> '1.7'
        if "." not in tok and len(tok) >= 2:
            return float(f"{tok[0]}.{tok[1:]}")
        # Single digit only (e.g. '2') -> '2.0'
        if "." not in tok:
            return float(f"{tok}.0")
        # Otherwise just float
        return float(tok)

    # 3) a:b format
    if ":" in s:
        a_raw, b_raw = s.split(":", 1)
        a = side_to_float(a_raw)
        b = side_to_float(b_raw)
        if a is None or b is None or a <= 0 or b <= 0:
            return ""

    # 4) Typical patterns without a colon
    else:
        only = re.sub(r"[^0-9.]", "", s)
        if only == "":
            return ""
        # '1.21' / '1.2' / '1.2.0' style
        m = re.fullmatch(r"1\.(\d+)(?:\.(\d+))?", only)
        if m:
            d1 = m.group(1)
            d2 = m.group(2)
            if d2 is not None:
                # 1.2.0 → 1 : 2.0
                rest = d1 + d2
                b = float(f"{rest[0]}.{rest[1:]}" if len(rest) >= 2 else f"{rest}.0")
            else:
                # 1.21 → 1 : 2.1 / 1.2 → 1 : 2.0
                b = float(f"{d1[0]}.{d1[1:]}" if len(d1) >= 2 else f"{d1}.0")
            a = 1.0
        # '5.1' → 5.1 : 1
        elif re.fullmatch(r"\d+\.\d+", only):
            a = float(only); b = 1.0
        # '0.xx' (ratio display) -> restore to a ratio as 1:X or X:1
        elif re.fullmatch(r"0\.\d+", only):
            p = float(only)
            if p <= 0 or p >= 1:
                return ""
            X = (1.0 - p) / p  # = E/I
            # Express with X>=1 (1:X or X:1)
            if X >= 1.0:
                a, b = 1.0, X
            else:
                a, b = 1.0 / X, 1.0
        # '17' → 1 : 1.7
        elif re.fullmatch(r"\d+", only):
            a = 1.0
            b = float(f"{only[0]}.{only[1:]}" if len(only) >= 2 else f"{only}.0")
        else:
            return ""

    # 5) From here, force "1:X or X:1", X with 1 decimal place, X>=1
    #    (scale the actual a:b so one side is 1 and the other is >=1)
    if a <= 0 or b <= 0:
        return ""
    if a <= b:
        X = b / a
        X = max(1.0, round(X, 1))
        return f"1:{X:.1f}"
    else:
        X = a / b
        X = max(1.0, round(X, 1))
        return f"{X:.1f}:1"

def _last_reading_settled(s):
    """Last non-NaN value of a breath's frames (in frame order); a value that appears only on the last frame is skipped."""
    v = s.dropna()
    if v.empty:
        return np.nan
    if len(v) >= 2 and v.index[-1] == s.index[-1] and (v == v.iloc[-1]).sum() == 1:
        return v.iloc[-2]
    return v.iloc[-1]


def Shift_Values(df, column):
    # Take the first 5 frames of the next breath number and get the mode
    mode_values = {}
    grouped = df.groupby('BreathNumber')

    for breath, group in grouped:
        # Take the first 5 frames of each breath (as-is if fewer than 5)
        first_values = group[column].head(5)
        # Compute the mode (use the first if there are several)
        mode_val = first_values.mode()
        mode_values[breath] = mode_val.iloc[0] if not mode_val.empty else first_values.iloc[0]

    # Get the mode of the next breath and shift
    next_values = pd.Series(mode_values).shift(-1)

    # Map the corresponding Real_ value to each row
    return df['BreathNumber'].map(next_values)


def most_frequent(series):
    mode_values = series.dropna().mode()
    return mode_values.iloc[0] if not mode_values.empty else None


# ---- Breath boundaries (0.2.0) ----
# A breath starts where the ventilator updates its per-breath displays. Cdyn and VTe are updated together at the start
# of each breath (checked on the NAVA sample: 46 of 46). A display update is a new value (or "***") that stays at least
# STABLE_SEC (shorter changes are misreads); updates of Cdyn and VTe within MERGE_SEC count as one boundary.
# Up to 0.1.x the boundaries came from Cdyn only and were then merged or split by I:E (CreateBreathNumber Steps 1.5-3.5),
# which merged real breaths whose Ti/Ttot was 0.5 or more (inverse-ratio I:E such as "6.3 : 1"). Breaths are no longer
# merged or split here; combining breaths is left to the combine_breaths module (same policy as the Hamilton engine).
BOUNDARY_KEYS = ("Cdyn", "VTe")
STABLE_SEC = 0.3
MERGE_SEC = 0.5
LONG_RATIO = 1.5        # Alert Two_Breath: at least this many times the moving average of breath lengths
SHORT_RATIO = 0.5       # Alert Short: at most this many times the moving average
LENGTH_WINDOW = 30      # breaths in the moving average


def _display_state(df, key):
    """Shown value of a display per frame: the number, "***" (no value), or None (not readable / abnormal frame, carried forward)."""
    val = pd.to_numeric(df[key], errors="coerce")
    raw = df.get(f"{key}_raw", pd.Series("", index=df.index)).fillna("").astype(str)
    stars = raw.str.contains(r"\*", regex=True) & ~raw.str.contains(r"\d", regex=True)
    state = pd.Series([f"{v:g}" if np.isfinite(v) else ("***" if st else None) for v, st in zip(val, stars)], index=df.index, dtype=object)
    if "Abnormal" in df:
        state[pd.to_numeric(df["Abnormal"], errors="coerce").fillna(0).to_numpy() == 1] = None
    return state.ffill()


def breath_boundaries(df, fps):
    """1 at the first frame of each breath (display update of Cdyn or VTe), else 0. df must be in frame order."""
    stable = max(1, round(STABLE_SEC * fps))
    merge = max(1, round(MERGE_SEC * fps))
    events = set()
    for key in BOUNDARY_KEYS:
        if key not in df:
            continue
        st = _display_state(df, key).to_numpy(object)
        n, cur, i = len(st), None, 0
        while i < n:
            v = st[i]
            if v is None:
                i += 1
                continue
            if cur is None:
                cur = v
                i += 1
                continue
            if v != cur:
                j = i
                while j < n and st[j] == v:
                    j += 1
                if j - i >= stable or j == n:
                    events.add(i)
                    cur = v
                i = j
                continue
            i += 1
    marks = np.zeros(len(df), int)
    last = None
    for e in sorted(events):
        if last is not None and e - last < merge:
            continue
        marks[e] = 1
        last = e
    return marks


VOLUME_MAX_ML = 5000.0         # 0.2.1: VTe / VTi reads above this are not values (physiologically impossible)


# Alert Disconnect (0.2.1). While the circuit is open (Servo-U 1-h video, video 1423-1459 s and 1521-1551 s) the
# breath's own VTe is 0.0-0.3 ml, VTi about 2050 ml (leak about 100%), and Cdyn shows "***". Single short or interrupted
# breaths also show VTe about 0, so only runs of breaths are flagged.
DISCONNECT_VTE_MAX = 5.0       # ml, the breath's own VTe (Real_VTe)
DISCONNECT_VTI_MIN = 1000.0    # ml, the breath's VTi (Real_VTi)
DISCONNECT_CDYN_STARS = 0.5    # more than this share of the breath's frames show Cdyn "***"
DISCONNECT_MIN_RUN = 3         # breaths in a row (a single breath that does not match inside a run is filled in)


def breath_disconnect(df, df_breath):
    """Breath numbers with Alert Disconnect. df: frames (with Real_VTe, Real_VTi, Cdyn_raw); df_breath: one row per breath."""
    if not {"Real_VTe", "Real_VTi", "Cdyn_raw"} <= set(df.columns) or df_breath.empty:
        return set()
    raw = df["Cdyn_raw"].fillna("").astype(str)
    stars = (raw.str.contains(r"\*", regex=True) & ~raw.str.contains(r"\d", regex=True)).groupby(df["BreathNumber"]).mean()
    b = df_breath.sort_values("BreathNumber")
    bn = b["BreathNumber"].to_numpy()
    hit = ((pd.to_numeric(b["Real_VTe"], errors="coerce") <= DISCONNECT_VTE_MAX)
           & (pd.to_numeric(b["Real_VTi"], errors="coerce") >= DISCONNECT_VTI_MIN)
           & (b["BreathNumber"].map(stars).fillna(0) > DISCONNECT_CDYN_STARS)).to_numpy()
    filled = hit.copy()
    filled[1:-1] |= hit[:-2] & hit[2:]
    out, start = set(), None
    for i, h in enumerate(np.r_[filled, False]):
        if h and start is None:
            start = i
        elif not h and start is not None:
            if i - start >= DISCONNECT_MIN_RUN:
                out.update(int(x) for x in bn[start:i])
            start = None
    return out


def breath_alerts(breath_numbers):
    """Alert per breath from its length: Fragment (first/last breath, cut by the recording), Two_Breath, Short."""
    lengths = pd.Series(breath_numbers).value_counts().sort_index()
    alerts = {b: [] for b in lengths.index}
    if len(lengths):
        alerts[lengths.index[0]].append("Fragment")
        alerts[lengths.index[-1]].append("Fragment")
        avg = lengths.rolling(window=max(1, min(LENGTH_WINDOW, len(lengths))), min_periods=1, center=True).mean()
        for b in lengths.index[lengths >= LONG_RATIO * avg]:
            alerts[b].append("Two_Breath")
        for b in lengths.index[lengths <= SHORT_RATIO * avg]:
            alerts[b].append("Short")
    return {b: ",".join(dict.fromkeys(a)) for b, a in alerts.items()}




# Cleaning OCRed text. Especially deleting noise like 2nd "."
def clean_decimal_number(text):
    """
    Extract and clean up digits and decimal points in the text and return a float or int.

    Main processing:
    - Remove all characters other than digits and decimal points
    - If there are two or more decimal points, keep only the first and remove the rest
    - Remove a trailing decimal point
    - Convert to float or int depending on the cleaned string
    - Return None if there are no digits or the value is invalid

    Parameters
    ----------
    text : str
        String obtained by OCR etc. (e.g. "6.2mg", "12..3", "abc7.89g")

    Returns
    -------
    float or int or None
        The successfully extracted and converted number (float or int). None if conversion fails.

    Tests
    -------
    ✅ Unit Test: test_valid_float()
    ✅ Unit Test: test_valid_int()
    ✅ Unit Test: test_invalid()
    """

    # delete characters other than number and decimal point
    text_cleaned = re.sub(r"[^\d.]", "", text)

    # delete second decimal points
    if text_cleaned.count(".") > 1:
        first_dot_index = text_cleaned.find(".")
        text_cleaned = text_cleaned[:first_dot_index + 1] + text_cleaned[first_dot_index + 1:].replace(".", "")

    # delete last decimal point
    if text_cleaned.endswith("."):
        text_cleaned = text_cleaned[:-1]

    # convert to int or float
    if "." in text_cleaned:
        return float(text_cleaned)  # to float
    elif text_cleaned.isdigit():
        return int(text_cleaned)  # to int
    else:
        return None  # invalid data




    
def VTi_Imputation(df):
    n = len(df)
    for i in range(1,n-1):
        if df.loc[i, "VTi"] != df.loc[i+1, "VTi"]:
            if df.loc[i, "VTi"] != df.loc[i-1, "VTi"]:
                if df.loc[i-1,"VTi"] == df.loc[i+1, "VTi"]:
                    df.loc[i, "VTi"] = df.loc[i-1, "VTi"]
    return df



# ===== Helpers for IE_Ratio (added right above parse_IE_ratio) =====
import re, unicodedata

def _fix_round_X(x: float) -> float:
    """Round X to 1 decimal place. X>=10 is treated as a decimal-point misread and divided by 10 once; X<1 becomes 1.0."""
    x = float(x)
    if x >= 10.0: x = x / 10.0      # e.g. 21 -> 2.1
    if x < 1.0:  x = 1.0
    return round(x + 1e-8, 1)

def _preclean_space_comma_to_colon(s: str) -> str:
    """Whitespace/comma -> colon, full-width -> half-width, collapse consecutive colons, strip edge colons."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = (s.replace("・", ".").replace("_", ".").replace("：", ":")
           .replace(",", ":").replace("，", ":").replace("、", ":"))
    s = s.replace("|", " ")                      # treat '|' from raw-text joining as whitespace (just in case)
    # Do not break double dots here (first just remove extra whitespace)
    if re.search(r"\d+\.\d+\.\d+", s):
        s = re.sub(r"\s+", "", s)
        return s.upper()
    s = re.sub(r"\s+", ":", s)                   # runs of whitespace -> colon
    s = re.sub(r":{2,}", ":", s)                 # collapse consecutive colons
    s = s.replace(":.:", ":")                    # '1 . 2.2' → '1:2.2'
    s = s.strip(":")                             # strip leading/trailing colons
    s = re.sub(r"\s+", "", s)                    # just in case
    return s.upper()

def _normalize_two_dot(s_pre: str) -> str:
    """Normalize a.b.c -> 1:X / X:1 (X>=1, 1 decimal place). Returns '' if not applicable."""
    if not s_pre or ":" in s_pre:
        return ""
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", s_pre)
    if not m:
        return ""
    A = float(m.group(1))
    B = float(f"{m.group(2)}.{m.group(3)}")
    if A <= 0 or B <= 0:
        return ""
    if A <= B:
        return f"1:{_fix_round_X(B/A):.1f}"
    else:
        return f"{_fix_round_X(A/B):.1f}:1"

def _normalize_colon_leading_dot(s_pre: str) -> str:
    """Restore 1:.xx / 1:.x.x... / .xx:1 / .x.x...:1 to 1:X / X:1. Returns '' if not applicable."""
    if not s_pre or ":" not in s_pre:
        return ""
    # 1:.xx / 1:.x.x...
    m = re.fullmatch(r"\s*1\s*:\s*\.(\d+(?:\.\d+)*)\s*", s_pre)
    if m:
        d = m.group(1)
        digits = re.sub(r"\D", "", d)           # e.g. "2.1" -> "21"
        if digits == "": 
            return ""
        X = float(f"{digits[0]}.{digits[1:]}") if len(digits) >= 2 else float(f"{digits}.0")
        return f"1:{_fix_round_X(X):.1f}"

    # .xx:1 / .x.x...:1
    m = re.fullmatch(r"\s*\.(\d+(?:\.\d+)*)\s*:\s*1\s*", s_pre)
    if m:
        d = m.group(1)
        digits = re.sub(r"\D", "", d)
        if digits == "":
            return ""
        X = float(f"{digits[0]}.{digits[1:]}") if len(digits) >= 2 else float(f"{digits}.0")
        return f"{_fix_round_X(X):.1f}:1"

    return ""


def _passthrough_if_canonical(s_pre: str) -> str:
    """If already in 1:X / X:1 form, correct and round X and return it (X>=1, 1 digit). Returns '' if not applicable."""
    if not s_pre or ":" not in s_pre:
        return ""
    # Safeguard when extra tokens are attached: look at only the first 2 tokens
    parts = s_pre.split(":")
    if len(parts) >= 3:
        s_pre = f"{parts[0]}:{parts[1]}"
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*", s_pre)
    if not m:
        return ""
    a = float(m.group(1)); b = float(m.group(2))
    if a == 1.0 and b >= 1.0:
        return f"1:{_fix_round_X(b):.1f}"
    if b == 1.0 and a >= 1.0:
        return f"{_fix_round_X(a):.1f}:1"
    return ""

def _normalize_one_dot_1xx(s_pre: str) -> str:
    """1.xx -> 1:X (X>=1, 1 digit). Returns '' if not applicable."""
    if not s_pre or ":" in s_pre:
        return ""
    m = re.fullmatch(r"1\.(\d+)$", s_pre)
    if not m:
        return ""
    d = m.group(1)
    X = float(f"{d[0]}.{d[1:]}") if len(d) >= 2 else float(f"{d}.0")
    return f"1:{_fix_round_X(X):.1f}"


# ===== Replaced from here: parse_IE_ratio =====
def parse_IE_ratio(ie_raw):
    """
    Convert IE_Ratio_raw -> IE_Ratio (numeric I/(I+E)).
    Spec:
      - A pure decimal (<1) is adopted as the ratio as-is ('.xx' is also interpreted as 0.xx)
      - Otherwise, only X:Y forms or decimals >1 (colon misrecognition) are normalized:
        preprocessing (whitespace/comma -> colon, etc.) -> double dot -> restore 1:.xx/.xx:1 -> already 1:X/X:1 -> 1.xx
        in that priority order, shaped into 1:X / X:1 (X>=1, 1 decimal place, X>=10 divided by 10 once), and a/(a+b) is returned.
    """
    s_pre = _preclean_space_comma_to_colon(ie_raw)
    if not s_pre:
        return 0.0

    # 1) A pure decimal (<1) is adopted as the ratio as-is ('.xx' is also 0.xx)
    t = re.sub(r"[^0-9.]", "", s_pre)
    if t:
        if t.startswith("."):
            t = "0" + t
        if re.fullmatch(r"0\.\d+", t):
            # 0.2.1: the Servo-U shows Ti/Ttot with two decimals ("0.30"); another length is a misread that lost or
            # added a digit ("0.2" for 0.30 or 0.22) and is left empty (the breath uses the other frames)
            if not re.fullmatch(r"0\.\d\d", t):
                return np.nan
            try:
                val = float(t)
                return val if 0.0 < val < 1.0 else 0.0
            except ValueError:
                return 0.0

    # 2) X:Y or decimal >1 -> minimal normalization to 1:X / X:1
    norm = _normalize_two_dot(s_pre)
    if not norm:
        norm = _normalize_colon_leading_dot(s_pre)
    if not norm:
        norm = _passthrough_if_canonical(s_pre)
    if not norm:
        norm = _normalize_one_dot_1xx(s_pre)

    if not norm or ":" not in norm:
        return 0.0

    a_str, b_str = norm.split(":", 1)
    try:
        a = float(a_str); b = float(b_str)
    except ValueError:
        return 0.0
    denom = a + b
    return (a / denom) if denom else 0.0

def _preprocess_for_edi(roi: np.ndarray) -> np.ndarray:
    """
    Preprocessing only for the Edi region:
      1) Contrast enhancement (mild)
      2) Smoothing (reduce noise while preserving edges)
      3) Otsu binarization (to follow exposure differences)
    Input: assumed uint8 single-channel (grayscale)
    """
    g = roi.copy()
    g = cv2.convertScaleAbs(g, alpha=2.0, beta=0)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    _, g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return g
