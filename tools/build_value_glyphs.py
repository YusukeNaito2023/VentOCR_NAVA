"""Build the digit templates (value_glyphs.npz) for reading the numeric ROIs by template matching (core._read_glyphs).

python tools/build_value_glyphs.py PACKAGE OUT.npz [FRAMES_PER_SEGMENT]

PACKAGE: an analysis result of this engine with Frames.csv and cfr/output_XXXXX.mov (segment videos).
One template set per ROI. Labels come from the OCR values in Frames.csv (EasyOCR). Frames are sampled evenly in each segment; a ROI is used only
when its number of digit blobs equals the number of digits of the label. Per digit, the blobs are averaged; examples
that correlate below 0.9 with their own average are dropped (mislabels) and the average is computed again.
Run it with the engine directory as the working directory's parent (it imports the engine's core helpers).
"""
import os
import re
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glyphs import KEYS_FOR_TEMPLATES, ROI_DIGIT_BLOBS, glyph_vector, TEMPLATE_H, TEMPLATE_W  # noqa: E402
import Settings  # noqa: E402


def label_digits(key, row):
    raw = str(row.get(f"{key}_raw", "")).strip()
    if key == "IE_Ratio":
        return re.sub(r"\D", "", raw) if re.fullmatch(r"\d+(\.\d+)?( ?: ?\d+(\.\d+)?)?", raw) else None
    return re.sub(r"\D", "", raw) if re.fullmatch(r"\d+(\.\d+)?", raw) else None


def main(pkg, out, per_seg=6):
    f = pd.read_csv(os.path.join(pkg, "Frames.csv"), low_memory=False)
    examples = {}
    for seg, g in f.groupby("source_file"):
        cap = cv2.VideoCapture(os.path.join(pkg, "cfr", seg.replace(".csv", ".mov")))
        picks = g.iloc[np.linspace(0, len(g) - 1, per_seg + 2, dtype=int)[1:-1]]
        for _, row in picks.iterrows():
            if int(row.get("Abnormal", 0) or 0):
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(row["iFrame"]))
            ok, img = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(cv2.resize(img, Settings.BASE_RESOLUTION, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            for key in KEYS_FOR_TEMPLATES:
                digits = label_digits(key, row)
                if not digits:
                    continue
                x, y, w, h = Settings.ROIs[key]
                blobs = ROI_DIGIT_BLOBS(gray[y:y + h, x:x + w])
                if blobs is None or len(blobs["digits"]) != len(digits):
                    continue
                for ch, (vec, aspect) in zip(digits, (glyph_vector(blobs, i) for i in blobs["digits"])):
                    examples.setdefault(key, {}).setdefault(ch, []).append((vec, aspect))
        cap.release()
    arrays = {"size": np.array([TEMPLATE_H, TEMPLATE_W])}
    for key, per_char in examples.items():
        chars, templates, aspects, counts = [], [], [], []
        for ch in sorted(per_char):
            vecs = np.stack([v for v, _ in per_char[ch]]); asp = np.array([a for _, a in per_char[ch]])
            mean = vecs.mean(0); mean /= np.linalg.norm(mean)
            keep = (vecs.reshape(len(vecs), -1) @ mean.ravel()) >= 0.9
            if not keep.any():
                continue
            mean = vecs[keep].mean(0); mean -= mean.mean(); mean /= np.linalg.norm(mean)
            chars.append(ch); templates.append(mean); aspects.append(float(np.median(asp[keep]))); counts.append(int(keep.sum()))
        arrays[f"templates_{key}"] = np.stack(templates).astype(np.float32)
        arrays[f"aspects_{key}"] = np.array(aspects, np.float32)
        arrays[f"chars_{key}"] = np.array(chars)
        arrays[f"counts_{key}"] = np.array(counts)
        print(key, dict(zip(chars, counts)))
    np.savez_compressed(out, **arrays)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 6)
