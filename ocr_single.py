#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tool that reads and displays single-image OCR (one .npy image) using the core.py logic.
- ROI coordinates use ROIs from Settings.py
- IE_Ratio, like initial_ocr, "stores the raw strings and confidences", while
  also showing, for reference, the value converted to a number by parse_IE_ratio in core.py
- The other keys (VTe/EdiPeak/Cdyn/Triggered) strictly implement read_text from core.py

Usage:
    python ocr_single.py /path/to/frame_00000003.npy --gpu  # GPU enabled
    python ocr_single.py /path/to/frame_00000003.npy        # CPU
"""

import argparse
import sys
import os
import numpy as np
import re
import cv2

# ===== Settings (ROI coordinates) =====
# Follows the user-provided Settings.py
ROIs = {
    "VTe":       [1139, 450, 55, 30],
    "EdiPeak":   [953,  518, 67, 31],
    "Cdyn":      [1110, 531, 164, 62],
    "Triggered": [843,  50,  49, 34],
    "IE_Ratio":  [1086, 243, 168, 44],
}
# ^ Source: faithfully mirrors ROIs in Settings.py (do not change)


# ===== Strict port of functions from core.py =====
# Faithfully mirrors read_text and parse_IE_ratio (+ their helpers) from core.py
# Source: core.py

def read_text(reader, roi):
    result = reader.readtext(roi)

    raw_texts = []
    confidences = []

    for res in result:
        _, text, score = res

        # Misrecognition correction (same as core.py)
        text = text.replace("_", ".").replace("・", ".").replace("I", "1").replace("：",":")

        raw_texts.append(text)
        confidences.append(score)

    # Text for display
    full_text = ", ".join(raw_texts)
    confidence_str = ", ".join([f"{s:.2f}" for s in confidences])

    # Remove everything except digits and decimal points -> ['33.', '4'] -> ['33.', '4']
    numeric_parts = [re.sub(r"[^0-9.:]", "", t) for t in raw_texts]
    numeric_parts = [t for t in numeric_parts if re.match(r"^\d+\.?\d*$", t)]  # keep only 0-9 or those containing a decimal

    # If a decimal point is already present, prefer it; otherwise treat as split parts and join
    if len(numeric_parts) == 1:
        cleaned_value = numeric_parts[0]
    elif len(numeric_parts) >= 2:
        # If numeric_parts[0] ends with a period, simply join
        if numeric_parts[0].endswith('.'):
            cleaned_value = numeric_parts[0] + numeric_parts[1]
        else:
            cleaned_value = numeric_parts[0] + '.' + numeric_parts[1]
    else:
        cleaned_value = ""

    return cleaned_value, full_text, confidence_str


# ===== parse_IE_ratio and its helpers (strictly mirrors core.py) =====
import unicodedata

def _fix_round_X(x: float) -> float:
    x = float(x)
    if x >= 10.0: x = x / 10.0
    if x < 1.0:  x = 1.0
    return round(x + 1e-8, 1)

def _preclean_space_comma_to_colon(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = (s.replace("・", ".").replace("_", ".").replace("：", ":")
           .replace(",", ":").replace("，", ":").replace("、", ":"))
    s = s.replace("|", " ")
    if re.search(r"\d+\.\d+\.\d+", s):
        s = re.sub(r"\s+", "", s)
        return s.upper()
    s = re.sub(r"\s+", ":", s)
    s = re.sub(r":{2,}", ":", s)
    s = s.replace(":.:", ":")
    s = s.strip(":")
    s = re.sub(r"\s+", "", s)
    return s.upper()

def _normalize_two_dot(s_pre: str) -> str:
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
    if not s_pre or ":" not in s_pre:
        return ""
    m = re.fullmatch(r"\s*1\s*:\s*\.(\d+(?:\.\d+)*)\s*", s_pre)
    if m:
        d = m.group(1)
        digits = re.sub(r"\D", "", d)
        if digits == "":
            return ""
        X = float(f"{digits[0]}.{digits[1:]}") if len(digits) >= 2 else float(f"{digits}.0")
        return f"1:{_fix_round_X(X):.1f}"

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
    if not s_pre or ":" not in s_pre:
        return ""
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
    if not s_pre or ":" in s_pre:
        return ""
    m = re.fullmatch(r"1\.(\d+)$", s_pre)
    if not m:
        return ""
    d = m.group(1)
    X = float(f"{d[0]}.{d[1:]}") if len(d) >= 2 else float(f"{d}.0")
    return f"1:{_fix_round_X(X):.1f}"

def parse_IE_ratio(ie_raw):
    s_pre = _preclean_space_comma_to_colon(ie_raw)
    if not s_pre:
        return 0.0

    t = re.sub(r"[^0-9.]", "", s_pre)
    if t:
        if t.startswith("."):
            t = "0" + t
        if re.fullmatch(r"0\.\d+", t):
            try:
                val = float(t)
                return val if 0.0 < val < 1.0 else 0.0
            except ValueError:
                return 0.0

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


def _ensure_uint8_gray(img: np.ndarray) -> np.ndarray:
    """Shape an array loaded from .npy into uint8 grayscale for EasyOCR input"""
    if img is None:
        raise ValueError("Image is None")
    arr = np.array(img)

    # 3ch/4ch → Gray
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        # Simple weighted average (no OpenCV needed)
        rgb = arr[..., :3].astype(np.float32)
        gray = (0.299 * rgb[..., 2] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 0])
        arr = gray
    elif arr.ndim == 2:
        pass
    else:
        # Force any other shape into 1 channel
        arr = arr.squeeze()
        if arr.ndim != 2:
            raise ValueError(f"Unknown array shape: {arr.shape}")

    # Scale adjustment
    if arr.dtype != np.uint8:
        # Estimate the value range and normalize to 0-255 (when needed)
        vmin, vmax = float(np.min(arr)), float(np.max(arr))
        if vmax > vmin:
            arr = (255.0 * (arr - vmin) / (vmax - vmin)).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)

    return arr


def _preprocess_for_edi(roi: np.ndarray) -> np.ndarray:
    # Input is uint8 single-channel
    g = roi.copy()

    # Contrast enhancement (a bit mild, starting from 2.0)
    g = cv2.convertScaleAbs(g, alpha=2.0, beta=0)

    # Smoothing (Gaussian tends to preserve character edges)
    g = cv2.GaussianBlur(g, (3, 3), 0)

    # Automatic binarization (Otsu): follows exposure/capture differences
    _, g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return g


def main():
    parser = argparse.ArgumentParser(description="Single-frame OCR using core.py logic")
    parser.add_argument("npy_path", help="Input .npy image (e.g. frame_00000003.npy)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU in EasyOCR")
    args = parser.parse_args()

    # Load the image
    if not os.path.exists(args.npy_path):
        print(f"File not found: {args.npy_path}", file=sys.stderr)
        sys.exit(1)

    img = np.load(args.npy_path, allow_pickle=True)
    img = _ensure_uint8_gray(img)

    # EasyOCR Reader
    from easyocr import Reader
    reader = Reader(["en"], gpu=args.gpu)
    print(f"[INFO] GPU: {args.gpu}")

    # OCR all ROIs in order
    print("=" * 80)
    print(f"Input: {args.npy_path}")
    print("=" * 80)

    results = {}

    for key, (x, y, w, h) in ROIs.items():
        # Crop the ROI (clamped to the image bounds)
        H, W = img.shape[:2]
        x0 = max(0, min(x, W - 1))
        y0 = max(0, min(y, H - 1))
        x1 = max(0, min(x + w, W))
        y1 = max(0, min(y + h, H))
        if x1 <= x0 or y1 <= y0:
            print(f"[WARN] ROI {key} is outside the image bounds: ({x},{y},{w},{h})")
            roi = img
        else:
            roi = img[y0:y1, x0:x1]

        roi_for_ocr = _preprocess_for_edi(roi) if key == "EdiPeak" else roi

        if key == "IE_Ratio":
            # Same as initial_ocr in core.py: keep raw text and confidence (value is an empty string)
            result = reader.readtext(roi)  # <- IE_Ratio uses the "raw" ROI as before
            raw_texts   = [text for (_, text, _) in result] if result else []
            confidences = [score for (_, _, score) in result] if result else []
            raw_join = ", ".join(raw_texts)
            conf_join = ", ".join(f"{s:.2f}" for s in confidences)
            ie_value = parse_IE_ratio(raw_join)
            results[key] = {
                "value": "",
                "raw": raw_join,
                "conf": conf_join,
                "ie_ratio_numeric": ie_value,
            }
        else:
            # EdiPeak is preprocessed; the others are used as-is
            value, raw, conf = read_text(reader, roi_for_ocr)
            results[key] = {"value": value, "raw": raw, "conf": conf}


    # Display
    print("\n--- OCR RESULTS (per ROI) ---")
    order = ["VTe", "EdiPeak", "Cdyn", "Triggered", "IE_Ratio"]  # same order as Settings
    for k in order:
        r = results.get(k, {})
        if k != "IE_Ratio":
            print(f"[{k}]")
            print(f"  cleaned_value : {r.get('value','')}")
            print(f"  raw_texts     : {r.get('raw','')}")
            print(f"  confidences   : {r.get('conf','')}")
        else:
            print(f"[{k}]")
            print(f"  value         : {r.get('value','')}" )  # empty (initial_ocr behavior)
            print(f"  raw_texts     : {r.get('raw','')}")
            print(f"  confidences   : {r.get('conf','')}")
            print(f"  parsed (I/(I+E)): {r.get('ie_ratio_numeric',0.0):.3f}  # from parse_IE_ratio")

    print("\nDone.")

if __name__ == "__main__":
    main()
