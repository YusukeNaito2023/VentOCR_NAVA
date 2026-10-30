"""0.2.1: read the axis labels of the waveform bands (the top value of each band, the bottom value of Pressure and Flow).

The Servo-U writes each band's scale as grey numbers at the left edge of the band ("25" / "-3", "60" / "-60", "650",
"20"). At BASE_RESOLUTION the digits are only 7-8 px tall, blurred, and a minus sign touches the next digit, so single
digits cannot be told apart reliably (6, 5, 9 and 0 look alike). Each label is instead compared as a whole with the
labels seen before (axis_labels.npz, made by tools/build_axis_labels.py): the same label always gives nearly the same
pixels (RMS difference about 1.5 gray levels), a different label differs by 7.6 or more, also when that label has no
template. A label is read only when its best template is within MAX_RMS and the next best is at least MIN_GAP further
away; otherwise (a label without a template, a pop-up, an unusual image) it is not read and the caller keeps its
estimate. Pixels of a waveform crossing the label (coloured) are left out of the comparison.

0.2.2: a label may have more than one template (recordings drawn slightly differently, e.g. Servo4 vs Servo-001); the
gap is then measured to the best template of a different label.

This module imports nothing from the engine (tools/build_axis_labels.py uses it directly).
"""
import os

import cv2
import numpy as np

# First row (y at BASE_RESOLUTION) of each label's text. The labels are centred at x 158.
LABEL_ROWS = {
    "Pressure_top": 93, "Pressure_bottom": 209,
    "Flow_top": 229, "Flow_bottom": 345,
    "Volume_top": 366,
    "Edi_top": 503,
}
X0, X1 = 140, 178              # label box columns (BASE_RESOLUTION)
ROWS_ABOVE, ROWS_BELOW = 2, 10  # label box rows around LABEL_ROWS
SHIFT = 1                      # the box is also compared shifted by up to this many pixels
MAX_RMS = 4.5                  # gray levels; the same label: 99% below 2.2, at most 3.3 (Servo-U 1-h video)
MIN_GAP = 3.0                  # the second-best template must be at least this much further
MAX_MASKED = 0.25              # at most this fraction of the box may be covered by a waveform
COLOR_SAT, COLOR_VAL = 60, 60  # a pixel with saturation and value at least these is part of a waveform


def label_box(img, slot, shift=0):
    """Gray label box (float32) and waveform mask of slot `slot` in a BASE_RESOLUTION BGR image.
    shift > 0 adds that many pixels on every side."""
    y = LABEL_ROWS[slot]
    c = img[y - ROWS_ABOVE - shift:y + ROWS_BELOW + shift, X0 - shift:X1 + shift]
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    colour = ((hsv[..., 1] >= COLOR_SAT) & (hsv[..., 2] >= COLOR_VAL)).astype(np.uint8)
    return cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32), cv2.dilate(colour, np.ones((3, 3), np.uint8)).astype(bool)


def normalized(gray, mask=None):
    """Label box minus its background level (median of the pixels not covered by a waveform)."""
    keep = gray if mask is None or not mask.any() else gray[~mask]
    return gray - np.median(keep)


class AxisLabelReader:
    def __init__(self, path):
        z = np.load(path)
        self.sets = {}
        for slot in LABEL_ROWS:
            if f"texts_{slot}" in z.files:
                self.sets[slot] = ([str(t) for t in z[f"texts_{slot}"]], z[f"templates_{slot}"].astype(np.float32))

    def read_slot(self, img, slot):
        """-> label value (float) or None."""
        if slot not in self.sets:
            return None
        texts, templates = self.sets[slot]
        h, w = templates.shape[1:]
        gray, mask = label_box(img, slot, SHIFT)
        best = np.full(len(texts), np.inf)
        for dy in range(2 * SHIFT + 1):
            for dx in range(2 * SHIFT + 1):
                g, m = gray[dy:dy + h, dx:dx + w], mask[dy:dy + h, dx:dx + w]
                if m.mean() > MAX_MASKED:
                    continue
                diff = (normalized(g, m) - templates)[:, ~m]
                best = np.minimum(best, np.sqrt((diff ** 2).mean(axis=1)))
        i = int(np.argmin(best))
        if not best[i] <= MAX_RMS:
            return None
        other = [d for t, d in zip(texts, best) if t != texts[i]]
        if other and min(other) - best[i] < MIN_GAP:
            return None
        return float(texts[i])

    def read(self, img):
        """BASE_RESOLUTION BGR image -> {slot: value} of the labels that could be read."""
        out = {}
        for slot in self.sets:
            v = self.read_slot(img, slot)
            if v is not None:
                out[slot] = v
        return out


def load_reader(engine_dir, file_name):
    path = os.path.join(engine_dir, file_name)
    return AxisLabelReader(path) if os.path.isfile(path) else None
