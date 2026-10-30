"""Build the axis-label templates (axis_labels.npz) for reading the waveform scales (axis_labels.py, 0.2.1).

1. python tools/build_axis_labels.py extract VIDEO BOXES.npz
   Takes the key frames of a Servo-U video (ffmpeg, read-only) and keeps the label boxes of every slot as post_ocr sees
   them: the frame is resized to BASE_RESOLUTION with INTER_AREA and stored as JPEG quality 95 (as core._keep_frame).
2. python tools/build_axis_labels.py cluster BOXES.npz OUT_DIR
   Groups the boxes of each slot without a waveform crossing them (RMS difference CLUSTER_RMS or less) and writes
   OUT_DIR/clusters.json and one picture per slot (OUT_DIR/<slot>.png, cluster numbers at the top left).
3. Look at the pictures and write LABELS.json: {"<slot>": {"<cluster number>": "<label text>", ...}, ...}. Leave out
   clusters that are not a clean label (pop-ups, a line through the label, a label being redrawn).
4. python tools/build_axis_labels.py build BOXES.npz OUT_DIR/clusters.json LABELS.json OUT.npz [EXISTING.npz]
   Each template is the average of its cluster. With EXISTING.npz (e.g. the engine's axis_labels.npz) its templates
   are kept and the new ones added. A new template of a label already there is added as another variant of it
   (0.2.2; e.g. a recording drawn slightly differently), unless it is within SAME_RMS of an existing one.
Run it from anywhere; it imports axis_labels.py and Settings.py from the engine directory.
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import axis_labels as al  # noqa: E402
import Settings  # noqa: E402

CLUSTER_RMS = 5.0
SAME_RMS = 3.0  # a new template this close to an existing one of the same label is not added


def extract(video, out):
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", video], capture_output=True, text=True, check=True)
    w, h = map(int, probe.stdout.strip().split(",")[:2])
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-skip_frame", "nokey", "-i", video, "-fps_mode", "passthrough",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    boxes = {slot: [] for slot in al.LABEL_ROWS}
    n = 0
    while True:
        buf = p.stdout.read(w * h * 3)
        if len(buf) < w * h * 3:
            break
        img = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        if (w, h) != Settings.BASE_RESOLUTION:
            img = cv2.resize(img, Settings.BASE_RESOLUTION, interpolation=cv2.INTER_AREA if w > Settings.BASE_RESOLUTION[0] else cv2.INTER_LINEAR)
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1], cv2.IMREAD_COLOR)
        for slot in al.LABEL_ROWS:
            boxes[slot].append(img[al.LABEL_ROWS[slot] - al.ROWS_ABOVE - al.SHIFT:al.LABEL_ROWS[slot] + al.ROWS_BELOW + al.SHIFT,
                                   al.X0 - al.SHIFT:al.X1 + al.SHIFT])
        n += 1
    p.wait()
    np.savez_compressed(out, **{slot: np.array(v) for slot, v in boxes.items()})
    print(f"{n} key frames -> {out}")


def _box(stored):
    """Stored box (with SHIFT margin) -> gray box and waveform mask without the margin."""
    s = al.SHIFT
    hsv = cv2.cvtColor(stored, cv2.COLOR_BGR2HSV)
    colour = (hsv[..., 1] >= al.COLOR_SAT) & (hsv[..., 2] >= al.COLOR_VAL)
    gray = cv2.cvtColor(stored, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return gray[s:gray.shape[0] - s, s:gray.shape[1] - s], colour[s:colour.shape[0] - s, s:colour.shape[1] - s]


def cluster(boxes_path, out_dir):
    z = np.load(boxes_path)
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    for slot in al.LABEL_ROWS:
        reps, members = [], []
        for i, stored in enumerate(z[slot]):
            gray, colour = _box(stored)
            if colour.any():
                continue
            v = al.normalized(gray)
            d = [float(np.sqrt(np.mean((v - r) ** 2))) for r in reps]
            if d and min(d) <= CLUSTER_RMS:
                members[int(np.argmin(d))].append(i)
            else:
                reps.append(v)
                members.append([i])
        out[slot] = members
        tiles = []
        for k, m in enumerate(members):
            g = np.mean([_box(z[slot][i])[0] for i in m], 0)
            t = cv2.resize(g, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
            cv2.putText(t, f"{k} ({len(m)})", (2, 12), cv2.FONT_HERSHEY_PLAIN, 0.9, 255, 1)
            tiles.append(np.pad(t, ((2, 2), (2, 2)), constant_values=90))
        if tiles:
            cols = 4
            th, tw = tiles[0].shape
            grid = np.full((-(-len(tiles) // cols) * th, cols * tw), 90, np.float32)
            for j, t in enumerate(tiles):
                grid[(j // cols) * th:(j // cols + 1) * th, (j % cols) * tw:(j % cols + 1) * tw] = t
            cv2.imwrite(os.path.join(out_dir, f"{slot}.png"), np.clip(grid, 0, 255).astype(np.uint8))
        print(slot, "clusters (size):", [len(m) for m in members])
    with open(os.path.join(out_dir, "clusters.json"), "w") as f:
        json.dump(out, f)


def build(boxes_path, clusters_path, labels_path, out, existing=None):
    z = np.load(boxes_path)
    clusters = json.load(open(clusters_path))
    labels = json.load(open(labels_path))
    sets = {}  # slot -> [(text, template)]
    if existing:
        e = np.load(existing)
        for slot in al.LABEL_ROWS:
            if f"texts_{slot}" in e.files:
                sets[slot] = list(zip([str(t) for t in e[f"texts_{slot}"]], e[f"templates_{slot}"]))
    for slot, named in labels.items():
        for k, text in named.items():
            boxes = [_box(z[slot][i])[0] for i in clusters[slot][int(k)]]
            new = al.normalized(np.mean(boxes, 0)).astype(np.float32)
            have = sets.setdefault(slot, [])
            if any(t == text and np.sqrt(np.mean((tpl - new) ** 2)) <= SAME_RMS for t, tpl in have):
                continue
            have.append((text, new))
    arrays = {}
    for slot, items in sets.items():
        items = sorted(items, key=lambda x: float(x[0]))
        arrays[f"texts_{slot}"] = np.array([t for t, _ in items])
        arrays[f"templates_{slot}"] = np.stack([tpl for _, tpl in items])
        print(slot, [t for t, _ in items])
    np.savez_compressed(out, **arrays)


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    {"extract": extract, "cluster": cluster, "build": build}[cmd](*args)
