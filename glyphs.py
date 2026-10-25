"""0.2.0: read the numeric ROIs by digit template matching (same idea as the Hamilton engine's Compliance reading).

The Servo-U draws each value with a fixed font, size and colour, and the digits never touch, so every digit is one
connected blob. A blob is resized to TEMPLATE_W x TEMPLATE_H, normalized (mean removed, unit length) and compared with
the templates of 0-9 of that ROI (value_glyphs.npz, made by tools/build_value_glyphs.py; one set per ROI, because the
fields are drawn at different sizes and a small digit is not a scaled-down large one). Small blobs are a decimal point (at the
baseline) or a colon (one at the middle, one at the baseline: I:E such as "1 : 2.3"); other small blobs (e.g. a small
star at the top right of Cdyn) are ignored.

read() returns (text, min_score) when every digit matches a template with at least MATCH_MIN, else None; the caller then
falls back to EasyOCR. A template read takes well under 1 ms; an EasyOCR call takes 7-30 ms.
This module imports nothing from the engine (tools/build_value_glyphs.py uses it directly).
"""
import os

import cv2
import numpy as np

TEMPLATE_H, TEMPLATE_W = 32, 20
MATCH_MIN = 0.9          # every digit must correlate with its best template at least this much
ASPECT_TOL = 0.15        # a digit's width/height may differ from its template's by at most this much
MIN_CONTRAST = 50        # ROI max minus median gray level; less = no text (blank display)
DIGIT_MIN_REL_H = 0.55   # blobs at least this fraction of the tallest blob are digits
SMALL_MAX_REL = 0.4      # blobs at most this fraction of the digit height (both ways) are dots / colon parts
KEYS_FOR_TEMPLATES = ("VTe", "EdiPeak", "Cdyn", "IE_Ratio", "Ppeak", "VTi")


def ROI_DIGIT_BLOBS(roi):
    """Gray ROI -> {"lab", "st", "digits": blob indices left to right, "tokens": [(x, "d"|"."|":", index)]}, or None."""
    roi = np.asarray(roi, np.uint8)
    if int(roi.max()) - int(np.median(roi)) < MIN_CONTRAST:
        return None
    _, bw = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, lab, st, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    st = st[1:]
    if not len(st):
        return None
    hmax = st[:, 3].max()
    digits = [i for i in range(len(st)) if st[i, 3] >= DIGIT_MIN_REL_H * hmax]
    if not digits:
        return None
    top = st[digits, 1].min()
    bottom = (st[digits, 1] + st[digits, 3]).max()
    H = bottom - top
    small = [i for i in range(len(st)) if i not in digits and st[i, 2] <= SMALL_MAX_REL * H and st[i, 3] <= SMALL_MAX_REL * H]
    base = [i for i in small if st[i, 1] + st[i, 3] >= bottom - 0.2 * H and st[i, 1] >= top + 0.5 * H]
    mid = [i for i in small if i not in base and top + 0.2 * H <= st[i, 1] + st[i, 3] / 2 <= top + 0.75 * H]
    tokens = [(st[i, 0], "d", i) for i in digits]
    used = set()
    for m in mid:
        for b in base:
            if b not in used and abs((st[m, 0] + st[m, 2] / 2) - (st[b, 0] + st[b, 2] / 2)) <= 0.3 * H:
                tokens.append((st[b, 0], ":", b)); used.add(b)
                break
    tokens += [(st[b, 0], ".", b) for b in base if b not in used]
    tokens.sort()
    return {"lab": lab, "st": st, "digits": sorted(digits, key=lambda i: st[i, 0]), "tokens": tokens}


def glyph_vector(blobs, i):
    """Normalized TEMPLATE_H x TEMPLATE_W image of blob i and its width/height."""
    x, y, w, h, _ = blobs["st"][i]
    v = cv2.resize((blobs["lab"][y:y + h, x:x + w] == i + 1).astype(np.float32), (TEMPLATE_W, TEMPLATE_H), interpolation=cv2.INTER_AREA)
    v -= v.mean()
    n = np.linalg.norm(v)
    return (v / n if n else v), w / h


class GlyphReader:
    def __init__(self, path):
        z = np.load(path)
        self.sets = {}
        for key in KEYS_FOR_TEMPLATES:
            if f"chars_{key}" in z:
                chars = [str(c) for c in z[f"chars_{key}"]]
                self.sets[key] = (chars, z[f"templates_{key}"].reshape(len(chars), -1), z[f"aspects_{key}"])

    def read(self, roi, key):
        """Gray ROI of `key` -> (text, min_score) or None (not readable by the templates: use EasyOCR)."""
        if key not in self.sets:
            return None
        chars, templates, aspects = self.sets[key]
        blobs = ROI_DIGIT_BLOBS(roi)
        if blobs is None:
            return None
        width = blobs["lab"].shape[1]
        st = blobs["st"]
        text, worst = "", 1.0
        for _, kind, i in blobs["tokens"]:
            if kind != "d":
                text += kind
                continue
            if st[i, 0] == 0 or st[i, 0] + st[i, 2] >= width:     # digit cut by the ROI edge
                return None
            v, aspect = glyph_vector(blobs, i)
            score = templates @ v.ravel()
            score[np.abs(aspects - aspect) > ASPECT_TOL] = -1
            k = int(np.argmax(score))
            if score[k] < MATCH_MIN:
                return None
            worst = min(worst, float(score[k]))
            text += chars[k]
        if text.count(".") > 1 + text.count(":") or text.count(":") > 1 or text[0] in ".:" or text[-1] in ".:":
            return None
        return text, worst


def load_reader(engine_dir, file_name):
    path = os.path.join(engine_dir, file_name)
    return GlyphReader(path) if os.path.isfile(path) else None
