#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
単体OCR（1枚の .npy 画像）を core.py のロジックで読み取り・表示するツール。
- ROI 座標は Settings.py の ROIs を使用
- IE_Ratio は initial_ocr と同様に「生文字列と信頼度を保存」しつつ、
  参考として core.py の parse_IE_ratio で数値化した値も表示
- それ以外のキー（VTe/EdiPeak/Cdyn/Triggered）は core.py の read_text を厳密実装

使い方:
    python ocr_single.py /path/to/frame_00000003.npy --gpu  # GPU有効
    python ocr_single.py /path/to/frame_00000003.npy        # CPU
"""

import argparse
import sys
import os
import numpy as np
import re
import cv2

# ===== Settings（ROI座標） =====
# ユーザー提供の Settings.py に準拠
ROIs = {
    "VTe":       [1139, 450, 55, 30],
    "EdiPeak":   [953,  518, 67, 31],
    "Cdyn":      [1110, 531, 164, 62],
    "Triggered": [843,  50,  49, 34],
    "IE_Ratio":  [1086, 243, 168, 44],
}
# ↑ 出典: Settings.py の ROIs を忠実に反映（変更しない）  :contentReference[oaicite:2]{index=2}


# ===== core.py の関数を厳密移植 =====
# read_text と parse_IE_ratio（＋その補助）を core.py から忠実に反映
# 出典: core.py  :contentReference[oaicite:3]{index=3}

def read_text(reader, roi):
    result = reader.readtext(roi)

    raw_texts = []
    confidences = []

    for res in result:
        _, text, score = res

        # 誤認識補正（core.py と同一）
        text = text.replace("_", ".").replace("・", ".").replace("I", "1").replace("：",":")

        raw_texts.append(text)
        confidences.append(score)

    # 表示用テキスト
    full_text = ", ".join(raw_texts)
    confidence_str = ", ".join([f"{s:.2f}" for s in confidences])

    # 数値・小数点以外の除去 → ['33.', '4'] → ['33.', '4']
    numeric_parts = [re.sub(r"[^0-9.:]", "", t) for t in raw_texts]
    numeric_parts = [t for t in numeric_parts if re.match(r"^\d+\.?\d*$", t)]  # 0-9または小数含むもののみ残す

    # 小数点がすでに含まれていればそれを優先、それ以外は分割扱いで結合
    if len(numeric_parts) == 1:
        cleaned_value = numeric_parts[0]
    elif len(numeric_parts) >= 2:
        # numeric_parts[0] が末尾にピリオドを含んでいれば、単純に結合
        if numeric_parts[0].endswith('.'):
            cleaned_value = numeric_parts[0] + numeric_parts[1]
        else:
            cleaned_value = numeric_parts[0] + '.' + numeric_parts[1]
    else:
        cleaned_value = ""

    return cleaned_value, full_text, confidence_str


# ===== parse_IE_ratio とそのヘルパ（core.py を厳密反映） =====
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
    """.npyで読み込んだ配列を EasyOCR 入力向けに uint8 グレースケールへ整形"""
    if img is None:
        raise ValueError("画像が None です")
    arr = np.array(img)

    # 3ch/4ch → Gray
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        # 簡易の加重平均（OpenCV不要）
        rgb = arr[..., :3].astype(np.float32)
        gray = (0.299 * rgb[..., 2] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 0])
        arr = gray
    elif arr.ndim == 2:
        pass
    else:
        # それ以外の形状は無理やり1chに
        arr = arr.squeeze()
        if arr.ndim != 2:
            raise ValueError(f"未知の配列形状です: {arr.shape}")

    # スケール調整
    if arr.dtype != np.uint8:
        # 値域を推定して 0-255 に正規化（必要時）
        vmin, vmax = float(np.min(arr)), float(np.max(arr))
        if vmax > vmin:
            arr = (255.0 * (arr - vmin) / (vmax - vmin)).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)

    return arr


def _preprocess_for_edi(roi: np.ndarray) -> np.ndarray:
    # 入力は uint8 1ch
    g = roi.copy()

    # コントラスト強調（少し弱め 2.0 から）
    g = cv2.convertScaleAbs(g, alpha=2.0, beta=0)

    # 平滑化（ガウシアンで文字エッジを保ちやすい）
    g = cv2.GaussianBlur(g, (3, 3), 0)

    # 自動二値化（Otsu）：露出や撮影差に追従
    _, g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return g


def main():
    parser = argparse.ArgumentParser(description="Single-frame OCR using core.py logic")
    parser.add_argument("npy_path", help="入力 .npy 画像（例: frame_00000003.npy）")
    parser.add_argument("--gpu", action="store_true", help="EasyOCR で GPU を使用する")
    args = parser.parse_args()

    # 画像読み込み
    if not os.path.exists(args.npy_path):
        print(f"File not found: {args.npy_path}", file=sys.stderr)
        sys.exit(1)

    img = np.load(args.npy_path, allow_pickle=True)
    img = _ensure_uint8_gray(img)

    # EasyOCR Reader
    from easyocr import Reader
    reader = Reader(["en"], gpu=args.gpu)
    print(f"[INFO] GPU: {args.gpu}")

    # すべての ROI を順に OCR
    print("=" * 80)
    print(f"Input: {args.npy_path}")
    print("=" * 80)

    results = {}

    for key, (x, y, w, h) in ROIs.items():
        # ROI 切り出し（範囲安全化）
        H, W = img.shape[:2]
        x0 = max(0, min(x, W - 1))
        y0 = max(0, min(y, H - 1))
        x1 = max(0, min(x + w, W))
        y1 = max(0, min(y + h, H))
        if x1 <= x0 or y1 <= y0:
            print(f"[WARN] ROI {key} が画像範囲外です: ({x},{y},{w},{h})")
            roi = img
        else:
            roi = img[y0:y1, x0:x1]

        roi_for_ocr = _preprocess_for_edi(roi) if key == "EdiPeak" else roi

        if key == "IE_Ratio":
            # core.py の initial_ocr と同様：生文字・信頼度を保持（値は空文字）
            result = reader.readtext(roi)  # ← IE_Ratioは従来どおり「生」ROI
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
            # EdiPeak は前処理済み、その他はそのまま
            value, raw, conf = read_text(reader, roi_for_ocr)
            results[key] = {"value": value, "raw": raw, "conf": conf}


    # 表示
    print("\n--- OCR RESULTS (per ROI) ---")
    order = ["VTe", "EdiPeak", "Cdyn", "Triggered", "IE_Ratio"]  # Settings と同順
    for k in order:
        r = results.get(k, {})
        if k != "IE_Ratio":
            print(f"[{k}]")
            print(f"  cleaned_value : {r.get('value','')}")
            print(f"  raw_texts     : {r.get('raw','')}")
            print(f"  confidences   : {r.get('conf','')}")
        else:
            print(f"[{k}]")
            print(f"  value         : {r.get('value','')}" )  # 空（initial_ocr仕様）
            print(f"  raw_texts     : {r.get('raw','')}")
            print(f"  confidences   : {r.get('conf','')}")
            print(f"  parsed (I/(I+E)): {r.get('ie_ratio_numeric',0.0):.3f}  # parse_IE_ratioより")

    print("\nDone.")

if __name__ == "__main__":
    main()
