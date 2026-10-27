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


VARIANT_MIN_SCORE = 0.7     # a failing digit is labelled with its best template only if it scores at least this much
VARIANT_MIN_MARGIN = 0.15   # ... and beats the second-best character by at least this much
VARIANT_MIN_COUNT = 10      # a variant template needs at least this many examples


def add_variants(npz_in, key, roi_dir, out):
    """Add a second template for digits the templates of `key` do not read (0.2.1).

    roi_dir: gray images of the ROI of `key` at BASE_RESOLUTION, cut the same way as the engine does (INTER_AREA, then
    core._GRAY_WEIGHTS). A digit whose best score is below glyphs.MATCH_MIN is labelled with its best character when that
    score is at least VARIANT_MIN_SCORE and VARIANT_MIN_MARGIN above the next character; per character the examples are
    averaged (examples below 0.9 to the average are dropped) and the average is added as an extra template of that
    character. Check the labels on images before using the result (the montage is written next to `out`).
    """
    from glyphs import GlyphReader, MATCH_MIN
    import glob
    reader = GlyphReader(npz_in)
    chars, templates, aspects = reader.sets[key]
    examples = {}
    for fn in sorted(glob.glob(os.path.join(roi_dir, "*.png"))):
        blobs = ROI_DIGIT_BLOBS(cv2.imread(fn, cv2.IMREAD_GRAYSCALE))
        if blobs is None:
            continue
        for _, kind, i in blobs["tokens"]:
            if kind != "d":
                continue
            vec, aspect = glyph_vector(blobs, i)
            score = templates @ vec.ravel()
            order = np.argsort(-score)
            best, second = score[order[0]], score[order[1]]
            ok_aspect = np.abs(aspects - aspect) <= 0.15
            if (score[ok_aspect].max() if ok_aspect.any() else -1) >= MATCH_MIN:
                continue
            if best >= VARIANT_MIN_SCORE and best - second >= VARIANT_MIN_MARGIN:
                examples.setdefault(chars[order[0]], []).append((vec, aspect))
    z = dict(np.load(npz_in))
    new_t, new_a, new_c, new_n, tiles = [], [], [], [], []
    for ch in sorted(examples):
        vecs = np.stack([v for v, _ in examples[ch]]); asp = np.array([a for _, a in examples[ch]])
        mean = vecs.mean(0); mean /= np.linalg.norm(mean)
        keep = (vecs.reshape(len(vecs), -1) @ mean.ravel()) >= 0.9
        if keep.sum() < VARIANT_MIN_COUNT:
            continue
        mean = vecs[keep].mean(0); mean -= mean.mean(); mean /= np.linalg.norm(mean)
        new_t.append(mean); new_a.append(float(np.median(asp[keep]))); new_c.append(ch); new_n.append(int(keep.sum()))
        for v in vecs[keep][:12]:
            tiles.append(((v - v.min()) / (np.ptp(v) or 1) * 255).astype(np.uint8))
        tiles += [np.zeros((TEMPLATE_H, TEMPLATE_W), np.uint8)] * (-len(tiles) % 12)
    print(key, "variants:", dict(zip(new_c, new_n)))
    if new_t:
        z[f"templates_{key}"] = np.concatenate([z[f"templates_{key}"], np.stack(new_t).astype(np.float32)])
        z[f"aspects_{key}"] = np.concatenate([z[f"aspects_{key}"], np.array(new_a, np.float32)])
        z[f"chars_{key}"] = np.concatenate([z[f"chars_{key}"], np.array(new_c)])
        z[f"counts_{key}"] = np.concatenate([z[f"counts_{key}"], np.array(new_n)])
        rows = [np.hstack(tiles[k:k + 12]) for k in range(0, len(tiles), 12)]
        cv2.imwrite(os.path.splitext(out)[0] + f"_{key}_variants.png", cv2.resize(np.vstack(rows), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
    np.savez_compressed(out, **z)


if __name__ == "__main__":
    # python tools/build_value_glyphs.py PACKAGE OUT.npz [FRAMES_PER_SEGMENT]
    # python tools/build_value_glyphs.py --add-variants IN.npz KEY ROI_DIR OUT.npz
    if sys.argv[1] == "--add-variants":
        add_variants(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 6)
