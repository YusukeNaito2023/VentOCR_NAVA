import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import Global_Setting as GS
from . import Settings

# 進捗(overall)報告はベストエフォート。import 失敗時も本処理は継続する。
try:
    import progress_io
except Exception:
    progress_io = None


def _report_progress(user_id, analyze_id, *, message, current, total):
    """post_ocr 内サブ処理の進捗を overall に流す（失敗しても無視）。
    完了点(current>=total)はスロットルに潰されないよう必ず書き込む。"""
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
    """フレーム画像を読み込み、基準解像度(Settings.BASE_RESOLUTION)へ正規化する。
    解像度非依存化のため、ROI/Y_RANGES定義の基準解像度に全フレームを揃える。"""
    img = cv2.imread(path)
    if img is None:
        return None
    if (img.shape[1], img.shape[0]) != Settings.BASE_RESOLUTION:
        interp = cv2.INTER_AREA if img.shape[1] > Settings.BASE_RESOLUTION[0] else cv2.INTER_LINEAR
        img = cv2.resize(img, Settings.BASE_RESOLUTION, interpolation=interp)
    return img

# --- 欠損値補完とスムージングを行うかどうか ---

SMOOTHING = True
IMPUTATION = True
ABNORMAL_WORKERS = 8  # DetectAbnormalFramesのフレーム並列読込スレッド数（imreadがI/O律速のため）

def set_config(settings):
    global SMOOTHING, IMPUTATION, ABNORMAL_WORKERS
    SMOOTHING = settings.get("post_ocr", {}).get("smoothing", True)
    IMPUTATION = settings.get("post_ocr", {}).get("imputation", True)
    ABNORMAL_WORKERS = settings.get("post_ocr", {}).get("abnormal_workers", 8)

# --- 色と位置の定義 ---
COLOR_DEFINITIONS = {
    "Pressure": np.array([110, 163, 193]),
    "Flow": np.array([90, 123, 86]),
    "Volume": np.array([114, 118, 89]),
    "Edi": np.array([171, 105, 166])
}

# 波形パネルの校正バンド（縦軸）。Photoshopによる再計測値（BASE_RESOLUTION=1280x720 上で計測）。
#   各 (top_y, bot_y) はパネル上端線=スケール最大値, 下端線=スケール最小値に対応し、
#   そのまま _pixels_to_physical の校正アンカーになる（top=max, bot=min）。
#     Pressure: top=+40線, bot=-4線（0線 y≈208 と内挿一致を確認）
#     Flow:     top=+max線, bot=-max線（対称軸, 0線 y≈291 が中点と一致）
#     Volume:   top=スケール最大線, bot=0mL線
#     Edi:      top=スケール最大線, bot=0µV基準線
#   ※ 旧値は解像度変更等で個別修正されドリフトしていたため、全波形を実測で再校正した。
#   ※ get_difference_regions もこの定義を単一の真実源として参照する（重複定義を排除）。
Y_RANGES = {
    "Pressure": (86, 222),
    "Flow":     (222, 358),
    "Volume":   (359, 495),
    "Edi":      (496, 631),
}

COLOR_TOLERANCE = 40

# --- HSV(色相)ベースのマッチ用パラメータ ---
# v1.2 変更点:
#   v1.1 は BGR のユークリッド距離（許容 COLOR_TOLERANCE）で波形色を判定していたが、
#   高解像度動画(1920x1080)を 1280x720 へ縮小する際、急峻な立ち上がり(ほぼ垂直な細線)が
#   アンチエイリアスで暗く混色し、BGR距離が許容を超えて NaN 欠損していた（立ち上がり検出不良）。
#   色相(Hue)は明度・彩度の低下に強く、機種差・縮小ボケがあっても安定するため、
#   Hue を主判定に切り替える。彩度/明度は背景・文字の除外にのみ用いる。

# 各波形の目標 Hue（OpenCV基準 0-180）。COLOR_DEFINITIONS から自動導出し v1.1 と色定義を共有する。
def _bgr_to_hue(bgr):
    px = np.uint8([[list(np.asarray(bgr, dtype=np.uint8))]])
    return int(cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0, 0])

TARGET_HUES = {name: _bgr_to_hue(bgr) for name, bgr in COLOR_DEFINITIONS.items()}

HUE_TOLERANCE = 12      # 色相の許容差（循環距離, 0-180）
SAT_MIN = 40            # これ未満の彩度は背景/灰色文字とみなし除外
VAL_MIN = 60            # これ未満の明度は背景とみなし除外
VAL_MIN_LEFT = 45       # 左端(140-175)はラベル領域で波形が暗いため明度条件を緩める
# 白色（スイープカーソルの先端）フォールバック用
WHITE_SAT_MAX = 45
WHITE_VAL_MIN = 140
BORDER_ROWS = 2         # バンド上端から除外する行数（パネル境界線）


# ============================================================================
#  軸スケール（縦軸目盛）の推定
#  パネル左上の軸の数字は 1280x720 換算で高さ 7〜8px しかなく、OCR がほとんど当たらない
#  （80→30、100/110→読めない）。そこで画面右の大きな表示値と波形の画素から呼吸ごとに逆算する。
#    Pressure: Real_Ppeak(cmH2O) ↔ 波形のピーク画素。0 cmH2O の基線（薄い横線）からの高さで比例させる
#    Volume:   Real_VTi(ml) ↔ 波形のピーク画素（0 ml = バンド下端）
#    Flow:     表示値が無いので、Flow 画素の積分が Volume(ml) の増減に一致するように求める
#  求めた「バンド上端の値」を目盛りの刻み（SCALE_SNAP）に丸め、前後の呼吸の最頻値で外れ値を除く。
#  Auto Scaling で目盛りは呼吸ごとに変わる（2026-06-01 の動画では Flow が 60〜110、Volume が 550〜850 を
#  行き来した）ので、呼吸ごとに求める。計算は画素の四則演算だけなので全呼吸で行っても軽い。
#  Edi は従来どおり EdiPeak で校正する（_edi_scale_from_peak）。
# ============================================================================
SCALE_UNITS = {"Pressure": "cmH2O", "Flow": "L/min", "Volume": "mL", "Edi": "uV"}
# チャートの手動スケール調整(+/-)のステップ幅（Engine特有）。
SCALE_STEPS = {"Pressure": 5, "Flow": 10, "Volume": 100, "Edi": 5}


def scale_meta():
    """波形チャートの軸スケール用メタ（Engine特有）。post_ocr が Archive に scale_meta.json
    として保存し、Label(Flask/JS) が読んでステップ幅・単位・対称性を適用する。
    キーは wave_data に合わせて小文字。"""
    return {
        "steps":     {k.lower(): v for k, v in SCALE_STEPS.items()},
        "units":     {k.lower(): v for k, v in SCALE_UNITS.items()},
        # 対称軸(min≒-max)か。Flowのみ True（+/- で両端を対称に動かす）。
        "symmetric": {k.lower(): k == "Flow" for k in SCALE_UNITS},
    }


# 0 の基線（薄い灰色の横線）の y（BASE_RESOLUTION）。目盛りが 25/30/40 と変わっても動かない（3本の動画で確認）。
# 呼吸ごとに終了画像で ±ZERO_SEARCH px の範囲を探し、見つからなければこの値を使う。
ZERO_Y = {"Pressure": 208, "Flow": 291}
ZERO_SEARCH = 4
# バンド上端の値（目盛りの最大値）の刻み。実画面で確認した値: Pressure 25/30/40, Flow 50〜110, Volume 400〜850
SCALE_SNAP = {"Pressure": 5, "Flow": 10, "Volume": 50}
SNAP_TOLERANCE = 0.35       # 推定値と丸めた値の差が 刻み×これ を超える呼吸は推定に使わない（どちらとも言えない）
SCALE_SMOOTH = 5            # 前後この呼吸数の最頻値で外れ値（二段吸気など）を除く
SWEEP_RATE_DEFAULT = 81.6   # 掃引の速さ px/秒（3本の動画で 81.5〜81.7）。開始/終了画像の掃引位置から求められないとき用
# Flow の積分から求めた目盛りは実際より約 6% 小さく出る（実画面の目盛りとの比: 1.03〜1.14、中央値 1.06）
FLOW_GAIN = 1.06
FLOW_WINDOW = 6             # Volume の増減と Flow の積分を比べる区間の長さ（px）


def _zero_px(img, name):
    """0 の基線の位置（バンド下端からの画素数＝波形CSVのピクセル値の座標）。"""
    y1, y2 = Y_RANGES[name]
    y0 = ZERO_Y[name]
    if img is not None:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        best, best_val = None, 40.0       # 背景の明るさ（約30）より明るい灰色の行だけを基線とみなす
        for y in range(y0 - ZERO_SEARCH, y0 + ZERO_SEARCH + 1):
            row = gray[y, 180:880][hsv[y, 180:880, 1] < 40]   # 波形（彩度が高い）を除いた灰色の画素
            if len(row) > 300 and np.median(row) > best_val:
                best, best_val = y, float(np.median(row))
        if best is not None:
            y0 = best
    return float(y2 - y0)


def _peak_px(values):
    """波形のピーク画素（1% の外れを除いた最大値）。点が少なければ None。"""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, 99)) if len(v) >= 20 else None


def _top_from_value(value, peak_px, zero_px, name, min_height=10):
    """表示値 value が波形のピーク画素 peak_px に当たるとして、バンド上端の値を返す。"""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value) or value <= 0 or peak_px is None or peak_px - zero_px < min_height:
        return None
    top, bot = Y_RANGES[name]
    return value * (bot - top - zero_px) / (peak_px - zero_px)


def _flow_top_from_volume(flow_px, volume_px, volume_scale, zero_px, rate):
    """Flow の積分 = Volume の増減 となるように、Flow のバンド上端の値（L/min）を求める。
    FLOW_WINDOW px ごとに「Volume の増減(ml)」と「Flow 画素の和 / 掃引の速さ」を集め、原点を通る直線で当てはめる。
    0 付近と上端に張り付いた Volume（呼気の終わり、目盛りからはみ出した吸気）は使わない。"""
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
        xs.append(seg.sum() / rate * 1000.0 / 60.0)   # (L/min/px を掛けると) ml
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
    """推定値を刻みに丸める。刻みのちょうど中間あたり（どちらとも言えない）なら None。"""
    if estimate is None or not np.isfinite(estimate) or estimate <= 0:
        return None
    snapped = max(step, round(estimate / step) * step)
    return float(snapped) if abs(estimate - snapped) <= SNAP_TOLERANCE * step else None


def _smooth_tops(values, width=SCALE_SMOOTH):
    """呼吸順の目盛り（None 可）を、前後 width 呼吸の最頻値に置き換える。同数なら自分の値、無ければ近い呼吸の値。"""
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
    # 前後に推定できた呼吸が無い呼吸は、直前（無ければ直後）の値で埋める
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
    """バンド上端の値と 0 の基線の位置から {min, max, unit}（バンド下端・上端の値）を作る。"""
    top, bot = Y_RANGES[name]
    h = bot - top
    return {"min": round(-zero_px * top_value / (h - zero_px), 2), "max": top_value, "unit": SCALE_UNITS[name]}


def _estimate_scales(records, rate):
    """BreathsToCSVs が集めた呼吸ごとの画素波形から {BreathNumber: {name: scale}} を求める。
    records: 呼吸順の dict（breath, waves, zero, ppeak, vti）"""
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
    """バンド下端=min, 上端=max の直線写像でピクセル値を物理単位へ変換する。"""
    top, bot = Y_RANGES[name]
    H = bot - top
    mn, mx = scale["min"], scale["max"]
    out = []
    for v in values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append(np.nan)
        else:
            # 2桁に丸める（CSV/JSONの肥大化＝Label描画遅延を防ぐ。画素分解能より十分細かい）
            out.append(round(mn + (float(v) / H) * (mx - mn), 2))
    return out


def _edi_scale_from_peak(edi_peak, edi_pixels, min_peak_px=8):
    """Edi の軸MAXはOCR困難なため、CSV済みの EdiPeak 値(µV)と波形ピーク画素で校正する。
    Edi波形ピクセルはバンド下端=0(=0µV)からの高さ。ピーク画素 peak_px が EdiPeak に対応する。
    → max(バンド上端の値) = EdiPeak * H / peak_px、min=0。
    返り値: {"min":0, "max":float, "unit":"uV"} / 校正不能なら None。"""
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
    if peak_px < min_peak_px:        # 振幅が小さすぎると校正が不安定
        return None
    top, bot = Y_RANGES["Edi"]
    H = bot - top
    return {"min": 0, "max": round(ep * H / peak_px, 2), "unit": SCALE_UNITS["Edi"]}


def _scale_frame_path(frames_root, source_file, iframe):
    sub = str(source_file).replace("output_", "").replace(".csv", "")
    return os.path.join(frames_root, sub, f"frame_{int(iframe):08d}.jpg")


def _hue_circular_dist(hue_array, target_hue):
    """色相の循環距離（0-90）を返す。"""
    d = np.abs(hue_array.astype(np.int16) - int(target_hue))
    return np.minimum(d, 180 - d)


# --- 色相(HSV)に基づく波形抽出 ---
# imageを渡すと波形の時系列データを返してくれる内部関数
def _extract_waveform(img, target_bgr, max_dist=40, waveform_name=None, y_offset=0):
    h, w, _ = img.shape
    timeseries = []
    target_hue = TARGET_HUES.get(waveform_name)
    # 列単位で BGR->HSV 変換するため、ROI全体を一度に変換しておく
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for x in range(w):
        column = img[:, x, :]
        hsv_col = hsv_img[:, x, :]
        H, S, V = hsv_col[:, 0], hsv_col[:, 1], hsv_col[:, 2]
        global_x = x + globals().get("x_start", 0)

        # 左端(140-175)はラベル/文字が重なるため、明度条件を緩めつつ灰色文字を明示的に除外する
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
        # バンド上端の数行はパネルの境界線。波形色と誤判定されて上端(=h)へ跳ねるため除外する
        valid[:BORDER_ROWS] = False

        # 判定: Edi はスイープ位置(847)以降を無効化（v1.1 と同一仕様）
        if waveform_name == "Edi" and 847 <= global_x:
            timeseries.append(np.nan)
        elif valid.any():
            idx = np.where(valid)[0]
            # 色相が最も近い画素を採用（同点は上側=先頭を優先）
            best = idx[np.argmin(hue_dist[idx])]
            timeseries.append(h - best)
        # 補助判定: 白色（スイープカーソル先端）で再挑戦
        elif waveform_name in ["Pressure", "Flow", "Edi"]:
            white = (S <= WHITE_SAT_MAX) & (V >= WHITE_VAL_MIN) & extra_ok
            white[:BORDER_ROWS] = False
            if white.any():
                timeseries.append(h - np.where(white)[0][0])
            else:
                timeseries.append(np.nan)
        else:
            timeseries.append(np.nan)
    return timeseries

# --- 画像から時系列データを抽出 ---
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
            waveforms[name] = series.rolling(window=10, center=True, min_periods=1).mean().to_numpy()
        waveforms[name] = series.to_numpy()
    return waveforms

# --- 差分領域の検出 ---
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

    # Y範囲（Pressure/Flow/Volume の3パネル）。Y_RANGES を単一の真実源として参照する
    # （以前はここに別値をハードコードしておりドリフトの温床だった）。
    y_ranges = [Y_RANGES["Pressure"], Y_RANGES["Flow"], Y_RANGES["Volume"]]

    # 画像読み込み
    img1 = _load_frame(image_path1)
    img2 = _load_frame(image_path2)

    # 各Y範囲に対する差分領域を抽出
    region_sets = []
    for y_start, y_end in y_ranges:
        regions = get_regions_for_y_range(img1, img2, y_start, y_end, x_start, x_end)
        region_sets.append(regions)

    # X軸上の重なりを調べるためにビットマスクを用意
    x_length = x_end + 1  # 893まで対応
    mask = np.zeros(x_length, dtype=int)

    # 各regionセットに重なりをカウント
    for regions in region_sets:
        temp = np.zeros(x_length, dtype=int)
        for x1, x2 in regions:
            temp[x1:x2 + 1] = 1
        mask += temp

    # 2つ以上のregionで重なっているXを取得
    overlap_coords = np.where(mask >= 2)[0]
    if len(overlap_coords) == 0:
        return []

    # X座標からregionに再変換
    merged = []
    start = overlap_coords[0]
    for i in range(1, len(overlap_coords)):
        if overlap_coords[i] > overlap_coords[i - 1] + 1:
            merged.append((start, overlap_coords[i - 1]))
            start = overlap_coords[i]
    merged.append((start, overlap_coords[-1]))

    return merged


# --- 差分領域の波形連結 ---
def merge_waveforms_across_regions(image_path, regions):
    merged = {}
    for x_start, x_end in sorted(regions, key=lambda r: -r[0]):
        waveforms = extract_waveform_timeseries(image_path, x_start, x_end)
        for key, values in waveforms.items():
            merged.setdefault(key, []).extend(values)
    return {k: np.array(v) for k, v in merged.items()}


# --- 終端フレームの掃引カーソル(先端)検出 ---
# v1.2 追加:
#   start/end 2フレーム差分は、区間が長い呼吸では現在の掃引に加えて「終端カーソルより右に
#   残っている前掃引の古い波形(残像)」まで拾い、複数波形に見える不具合があった。
#   終端フレームとその直前フレームの差分帯＝掃引先端(カーソル)位置であり、これより右の列は
#   前掃引の残像なので、抽出領域から落とすための基準xを返す。
def _sweep_cursor_from_imgs(img_e, img_b, x_lo=140, x_hi=893,
                            diff_thresh=25, smooth_kernel=15,
                            vote_panels=("Pressure", "Flow", "Volume"),
                            min_vote=2, cluster_gap=10, wrap_width=250):
    """2枚の画像(終端/直前, いずれも基準解像度)から掃引カーソルx(=先端＝新規描画の右端)を返す。
    検出不可なら None。画像は _load_frame 済み(BASE_RESOLUTION)である前提。

    カーソルは全波形パネルを縦断する縦の特徴なので、Pressure/Flow/Volume の各パネルで
    列ごとに平滑後差分が閾値超えかを投票し、min_vote(=2)票以上一致した列だけを採用する
    （get_difference_regions と同じ多数決。Edi はパネル固有マーカーがあり単一パネルで化けるため除外）。
    これで Edi マーカーや単一パネルのノイズに釣られず x_hi まで全幅探索できる。
    採用列をクラスタ化し、幅広クラスタ(掃引折り返し)があれば最右の幅広クラスタの左端、
    無ければ最右クラスタの右端(=先端)を返す。"""
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
    # 採用列をクラスタ化（cluster_gap 以内の列を1クラスタに束ねる）
    runs = []
    start = prev = xs[0]
    for x in xs[1:]:
        if x > prev + cluster_gap:
            runs.append((start, prev))
            start = x
        prev = x
    runs.append((start, prev))
    # 掃引折り返し: 幅広クラスタがあれば最右の幅広クラスタの左端＝カーソル
    wide = [r for r in runs if (r[1] - r[0]) > wrap_width]
    if wide:
        return x_lo + int(wide[-1][0])
    # 通常: 最右クラスタの右端＝掃引先端
    return x_lo + int(runs[-1][1])


def _detect_sweep_cursor(path_end, path_before, x_lo=140, x_hi=893,
                         diff_thresh=25, smooth_kernel=15):
    """終端フレームの掃引カーソルx(=先端＝最右の新規描画列)を返す。検出不可なら None。"""
    return _sweep_cursor_from_imgs(
        _load_frame(path_end), _load_frame(path_before),
        x_lo=x_lo, x_hi=x_hi, diff_thresh=diff_thresh, smooth_kernel=smooth_kernel,
    )


def annotate_sweep_lines(df_frames, frames_root, line_bgr=(0, 0, 255), thickness=2):
    """各フレームに掃引カーソル(Sweep Line)を縦線で描き込み、元のフレーム画像を上書き保存する。
    検出は基準解像度(BASE_RESOLUTION)で行い、描画は元解像度のフレームへ座標をスケールして引く
    （元解像度を保持＝Labelの元解像度ダウンロードと整合）。直前フレーム(無加工)との差分で先端xを
    検出し、波形帯(Pressure上端〜Edi下端)を縦断する線を引く。Detect Sweep Line オプション(既定OFF)
    有効時の診断用。検出不能フレームは無印のまま。描画したフレーム数を返す。"""
    if df_frames is None or getattr(df_frames, "empty", True):
        return 0
    base_w, base_h = Settings.BASE_RESOLUTION
    y0 = int(Y_RANGES["Pressure"][0])
    y1 = int(Y_RANGES["Edi"][1])
    ordered = df_frames.sort_values("Frame")
    prev_norm = None   # 直前フレームの基準解像度画像（検出専用・無加工）
    drawn = 0
    for _, fr in ordered.iterrows():
        path = _scale_frame_path(frames_root, fr.get("source_file"), fr.get("iFrame"))
        norm = _load_frame(path)               # 基準解像度（検出用）
        if norm is None:
            prev_norm = None
            continue
        cursor_x = _sweep_cursor_from_imgs(norm, prev_norm) if prev_norm is not None else None
        prev_norm = norm                       # 無加工のまま次フレーム検出に使う（線の混入なし）
        if cursor_x is None:
            continue
        orig = cv2.imread(path)                # 元解像度のまま（描画して上書き）
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
    return drawn                  # 先端＝最右の変化列


def _trim_regions_right_of_cursor(regions, cursor_x, left_edge=160, right_edge=872):
    """終端カーソルより右の領域(=前掃引の残像)を落とす。
    ただし真の折り返し(左右両端に別々の領域が存在)の場合は、右側が当該呼吸の正規描画なので
    トリムせず従来処理に委ねる。"""
    if cursor_x is None or not regions:
        return regions
    right_touch = any(r[1] >= right_edge for r in regions)
    left_touch = any(r[0] <= left_edge for r in regions)
    is_true_wrap = right_touch and left_touch and len(regions) >= 2
    if is_true_wrap:
        return regions
    trimmed = [(xs, min(xe, cursor_x)) for xs, xe in regions if xs <= cursor_x]
    return [(xs, xe) for xs, xe in trimmed if xe > xs]


# --- 掃引折り返し(Sweep wrap)対応の呼吸領域再構成 ---
# v1.2 追加:
#   差分は start/end 2フレームの「片方にしか波形が無い」全X位置を拾うため、掃引が1呼吸以上
#   進む/画面右端で折り返す呼吸では、別呼吸の古い残像断片まで領域に混ざり、連結すると
#   2つの山(プラトー)に見える不具合があった。
#   折り返しを検出した場合は「右端領域(折り返し直前に描画)→左端領域(折り返し直後に描画)」のみを
#   時系列順で採用し、中間の古い残像断片を破棄する（非折り返し時は従来通り全領域を使う）。
def _select_breath_regions(regions, left_edge=160, right_edge=872):
    if regions is None or len(regions) <= 1:
        return regions
    right = [r for r in regions if r[1] >= right_edge]   # 画面右端に接する領域
    left  = [r for r in regions if r[0] <= left_edge]    # 画面左端に接する領域
    if right and left:
        # 折り返し検出: 右端→左端の呼吸断片のみ採用し、中間の残像を捨てる
        sel = sorted(right, key=lambda r: r[0]) + sorted(left, key=lambda r: r[0])
        seen, out = set(), []
        for r in sel:
            key = (int(r[0]), int(r[1]))
            if key not in seen:
                seen.add(key)
                out.append(r)
        return out
    # 非折り返し: 従来通り（差分が局所的で残像混入が起きにくい）
    return regions


# --- 末尾に混入する「次呼吸の立ち上がり」をトリム ---
# v1.2 追加:
#   折り返し時の左端領域は、次呼吸の吸気立ち上がり開始位置まで差分が伸びることがあり、
#   再構成波形の末尾に次呼吸の立ち上がりが僅かに残る。Pressure波形で「一度ベースラインに
#   戻った後に再度立ち上がる(=次呼吸の吸気)」点を検出し、そこ以降を全波形から切り落とす。
#   1呼吸は通常 baseline→吸気→プラトー→呼気→baseline であり、2回目の立ち上がりは無いため、
#   2回目の立ち上がりを検出したときのみ保守的にトリムする。
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
    if amp < 15:  # 振幅が小さい（平坦/欠損）→ 判定不能なのでトリムしない
        return waveforms
    hi = base + 0.60 * amp
    lo = base + 0.30 * amp
    above = s > hi
    if not above.any():
        return waveforms
    first_up = int(np.argmax(above))                       # 1回目の吸気
    after = np.where(s[first_up:] < lo)[0]                  # その後ベースライン付近へ戻る
    if len(after) == 0:
        return waveforms
    back = first_up + int(after[0])
    nxt = np.where(s[back:] > hi)[0]                        # 再度立ち上がる（次呼吸の吸気）
    if len(nxt) == 0:
        return waveforms
    cut = back + int(nxt[0])
    # 立ち上がり開始（cut直前でベースライン付近を最後に通過した点）まで戻して切る
    pre = np.where(s[back:cut] <= lo)[0]
    if len(pre) > 0:
        cut = back + int(pre[-1]) + 1
    # 早すぎる切断（主呼吸を誤って切る）を防ぐガード
    if cut < n * 0.5:
        return waveforms
    return {k: np.asarray(v)[:cut] for k, v in waveforms.items()}

# --- Volume の非生理的V字谷(高→低→高)を除去 ---
# v1.2 追加:
#   急峻な立ち上がり/降下の列で本来の波形画素を取り逃すと、バンド下端(0ml線)付近の
#   迷光画素を誤採用し、孤立列だけ 0mL 付近に沈む（Breath 5,16 で発生）。
#   Volumeは1呼吸で単峰（上昇→下降）であり高→低→高のV字谷は非生理的なため、
#   両肩(前後の局所ピーク)に対して相対的に深く沈む孤立した谷を OCR誤検出として
#   NaN化し、線形補間で復元する。Flow/Pressure/Edi は双方向に振れるため対象外。
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
    base = np.nanpercentile(filled, 5)   # ベースライン近傍
    amp = peak - base
    if amp <= 0:
        return waveforms
    bad = np.zeros(n, dtype=bool)
    for i in range(n):
        if np.isnan(s[i]):
            continue
        l0, r1 = max(0, i - win), min(n, i + win + 1)
        lmax = np.nanmax(filled[l0:i]) if i > l0 else -np.inf      # 左肩
        rmax = np.nanmax(filled[i + 1:r1]) if i + 1 < r1 else -np.inf  # 右肩
        shoulder = min(lmax, rmax)
        if not np.isfinite(shoulder):
            continue
        # 両肩が十分高い谷(=単調な立ち上がり/降下では発火しない)で、
        # 谷底が肩に対し相対的に深く沈む孤立点のみを誤検出とみなす
        if (s[i] < trough_ratio * shoulder
                and shoulder - s[i] > prom_ratio * amp
                and shoulder > base + 0.1 * amp):
            bad[i] = True
    if not bad.any():
        return waveforms
    # 谷として除外した列だけを補間値で置き換える。元々のNaN欠損区間は
    # 触らず温存し（下流の補間/欠損扱いに委ねる）、副作用を最小化する。
    tmp = s.copy()
    tmp[bad] = np.nan
    interp = pd.Series(tmp).interpolate(method="linear", limit_direction="both").to_numpy()
    s_out = s.copy()
    s_out[bad] = interp[bad]
    return {**waveforms, name: s_out}


# --- メイン関数: 呼吸波形CSV生成 ---
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
        if not (os.path.exists(img_start) and os.path.exists(img_end)):
            continue
        regions = get_difference_regions(img_end, img_start)
        if not regions:
            continue
        regions = _select_breath_regions(regions)   # 掃引折り返し時の残像断片を除去
        waveforms = merge_waveforms_across_regions(img_end, regions)
        waveforms = _trim_trailing_breath(waveforms) # 末尾の次呼吸立ち上がりをトリム
        waveforms = _reject_volume_troughs(waveforms) # Volumeの非生理的V字谷を除去
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

def BreathsToCSVs(breaths, frames, frames_root, output_dir, user_id, overwrite=False, report_path=None, analyze_id=None, reader=None, settings=None):
    """
    各呼吸エピソードの Start/End 区間から差分領域を抽出し、波形（Pressure, Flow, Volume, Edi）を生成。
    {BreathNumber}.csv を出力し、加工後の breaths を返す。

    追加仕様:
        overwrite (bool): True なら既存 {BreathNumber}.csv を削除して上書き
        report_path (str|None): スキップ理由のCSV保存先（Noneなら output_dir/BreathCSV_Report.csv）
    """
    import os
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    os.makedirs(output_dir, exist_ok=True)
    if report_path is None:
        report_path = os.path.join(GS.Dir_Outputs(user_id), "BreathCSV_Report.csv")

    # ---- 準備：安全なマッピング用インデックスと範囲 ----
    # framesをIndexにして、それに対応するiframe, src (output_00000.csv)を取得
    # frameのmin/maxを取得しておき、この範囲でのみframes_idx_iframeなどにアクセス
    frames_idx_iframe = frames.set_index("Frame")["iFrame"]
    frames_idx_src    = frames.set_index("Frame")["source_file"]
    frame_min = int(frames["Frame"].min())
    frame_max = int(frames["Frame"].max())

    skip_logs = []
    summary_records = []

    # 軸スケール: 呼吸ごとに {BreathNumber: {name: {min,max,unit}}} を貯める。
    # 目盛りは前後の呼吸と合わせて決める（_estimate_scales）ので、まず全呼吸の画素波形を集めてから換算・書き出す。
    # reader は以前の軸の数字のOCR用の引数（互換のため残している。使わない）
    scale_enabled = (settings or {}).get("post_ocr", {}).get("scale", {}).get("enabled", True)
    breath_scales = {}
    records = []        # 呼吸順: {breath, waves(画素), zero, ppeak, vti, edi_peak, edi_full, df, out_path, ...}
    sweep_rates = []    # 開始/終了画像の掃引位置と時刻から求めた掃引の速さ（px/秒）
    # 診断用: 抽出に使った波形領域の左右端x と 開始/終端フレームの掃引カーソルx を呼吸ごとに貯める
    # {BreathNumber: (x_start, x_end, cursor_start, cursor_end)}。Sweep検出の妥当性をLabelで確認するため。
    breath_xspan = {}
    # 直前に成功した Edi 目盛り（EdiPeakが*** = NaN で校正不能な呼吸はこれを流用する）
    last_edi_scale = None


    # Start 側：Frame-6 をクリップしてからマップ（先頭で負になるのを回避）
    
    # Breathsに入っているFrame列を数値に念のため変換しbase_frameとしてシリーズ化。つまりBase＿Frameはそれぞれの呼吸開始Frame番号
    base_frame = pd.to_numeric(breaths["Frame"], errors="coerce")

    # base_frameにNanなどが混入していた場合はframe_minで置き換え、astypeで小数を整数に（17.0->17）、Clipで範囲外（例：baseframeが3で-6したら-3）の場合、frame_minに置き換え
    adjusted_frame_start = (base_frame - 6).fillna(frame_min).astype(int).clip(frame_min, frame_max)


    # 開始フレームの同定
    # Start_FileにはadjustedFrameStartで格納された値をキーにしてframes_idx_iframeを探しに行ってる。これでStartのiframeがstart_fileに入る
    # Start_Folderも同様の論理で最終的に[00000]のようなサブフォルダー名が入る
    breaths["Start_File"] = adjusted_frame_start.map(frames_idx_iframe)
    breaths["Start_Folder"] = (
        adjusted_frame_start.map(frames_idx_src)
        .str.replace("output_", "", regex=False)
        .str.replace(".csv",   "", regex=False)
    )

    # 終了フレームの同定
    # End 側：次呼吸の Frame をそのまま使う（最終行は End が作れないため除去）
    breaths["End_Frame"] = breaths["Frame"].shift(-1)
    breaths = breaths.drop(breaths.index[-1])

    adjusted_frame_end = pd.to_numeric(breaths["End_Frame"], errors="coerce").fillna(frame_max).astype(int)
    adjusted_frame_end = adjusted_frame_end.clip(frame_min, frame_max)

    breaths["End_File"] = adjusted_frame_end.map(frames_idx_iframe)
    breaths["End_Folder"] = (
        adjusted_frame_end.map(frames_idx_src)
        .str.replace("output_", "", regex=False)
        .str.replace(".csv",   "", regex=False)
    )
    breaths.drop(columns=["End_Frame"], inplace=True)

    # マップ失敗（NaN）は安全に落とす
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

    # 画像パスの作成（.\users\admin\Frames/00000/frame_00000039.jpgなど）
    breaths["Start_File_Name"] = (
        frames_root + "/" + breaths["Start_Folder"] + "/frame_" +
        breaths["Start_File"].astype(int).astype(str).str.zfill(8) + ".jpg"
    )
    breaths["End_File_Name"] = (
        frames_root + "/" + breaths["End_Folder"] + "/frame_" +
        breaths["End_File"].astype(int).astype(str).str.zfill(8) + ".jpg"
    )

    # 1呼吸目を除外
    breaths = breaths[breaths["BreathNumber"] != 1].reset_index(drop=True)

    # Map失敗による削除があるので、次のiterrowの前にindexをリセット
    breaths = breaths.reset_index(drop=True)

    # ---- 各呼吸を処理 ----
    _bt_total = len(breaths)
    for _bt_i, (_, row) in enumerate(tqdm(breaths.iterrows(), total=_bt_total), 1):
        if _bt_i % 20 == 0 or _bt_i == _bt_total:
            _report_progress(user_id, analyze_id,
                             message=f"Extracting breath CSVs ({_bt_i}/{_bt_total})",
                             current=_bt_i, total=_bt_total)
        # 境界値の取得と検証
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


        # FramesからBreathNumberでフィルターして当該データだけを抽出

        filtered = frames[frames["BreathNumber"] == breath_num].copy()

        # 空っぽならエラー
        if filtered.empty:
            skip_logs.append({
                "BreathNumber": breath_num, "Status": "SKIPPED",
                "Reason": "empty frame range", "Details": f"{start_iframe}-{end_iframe}",
            })
            continue

        # 一個でもAbnormal判定がついていればエラー
        filtered["Abnormal"] = pd.to_numeric(filtered["Abnormal"], errors="coerce")
        label_sum = int(filtered["Abnormal"].sum()) if not filtered["Abnormal"].isna().all() else 0
        if label_sum > 0:
            skip_logs.append({
                "BreathNumber": breath_num, "Status": "SKIPPED",
                "Reason": "abnormal frames", "Details": f"sum={label_sum}",
            })
            continue

        # 画像存在チェック
        img_start = str(row["Start_File_Name"])
        img_end   = str(row["End_File_Name"])
        if not (os.path.exists(img_start) and os.path.exists(img_end)):
            skip_logs.append({
                "BreathNumber": breath_num, "Status": "SKIPPED",
                "Reason": "image not found",
                "Details": f"start={os.path.exists(img_start)}, end={os.path.exists(img_end)}",
            })
            continue

        # 差分領域抽出
        regions = get_difference_regions(img_end, img_start)
        if not regions:
            skip_logs.append({
                "BreathNumber": breath_num, "Status": "SKIPPED",
                "Reason": "no difference regions", "Details": "",
            })
            continue

        # 終端カーソル右側の前掃引残像をトリム（直前フレームとの差分で先端を検出）。
        # 真の折り返し（左右別領域）の場合はトリムせず _select_breath_regions に委ねる。
        # 直前フレームは img_end と同じフォルダの (iframe-1) として導出する。
        cursor_x = None
        before_path = os.path.join(os.path.dirname(img_end),
                                   f"frame_{end_iframe - 1:08d}.jpg")
        if end_iframe - 1 >= 0 and os.path.exists(before_path):
            cursor_x = _detect_sweep_cursor(img_end, before_path)
            regions = _trim_regions_right_of_cursor(regions, cursor_x)

        regions = _select_breath_regions(regions)   # 掃引折り返し時の残像断片を除去

        # 開始フレーム側の掃引カーソルx（開始フレームと直前フレームの差分で検出）
        cursor_x_start = None
        before_start = os.path.join(os.path.dirname(img_start),
                                    f"frame_{start_iframe - 1:08d}.jpg")
        if start_iframe - 1 >= 0 and os.path.exists(before_start):
            cursor_x_start = _detect_sweep_cursor(img_start, before_start)

        # 診断用: 抽出領域の左右端x と 開始/終端フレームの検出カーソルx を記録（Breaths.csvへ出力）
        if regions:
            breath_xspan[breath_num] = (
                int(min(r[0] for r in regions)),
                int(max(r[1] for r in regions)),
                int(cursor_x_start) if cursor_x_start is not None else None,
                int(cursor_x) if cursor_x is not None else None,
            )

        # 波形抽出
        waveforms = merge_waveforms_across_regions(img_end, regions)
        waveforms = _trim_trailing_breath(waveforms) # 末尾の次呼吸立ち上がりをトリム
        waveforms = _reject_volume_troughs(waveforms) # Volumeの非生理的V字谷を除去
        length = len(next(iter(waveforms.values()), []))
        if length == 0:
            skip_logs.append({
                "BreathNumber": breath_num, "Status": "SKIPPED",
                "Reason": "empty waveform", "Details": "",
            })
            continue

        # 掃引の速さ（Flow の目盛りの推定に使う）。折り返しを含む呼吸と区間をまたぐ呼吸は使わない
        if cursor_x_start is not None and cursor_x is not None and cursor_x > cursor_x_start \
                and row["Start_Folder"] == row["End_Folder"] and end_iframe > start_iframe:
            sweep_rates.append((cursor_x - cursor_x_start) / ((end_iframe - start_iframe) / GS.FPS))

        # 上書き処理
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

        # CSV 出力（末尾 30 点カット）
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
            "summary": {
                "BreathNumber": breath_num,
                "Start_Frame_Image": img_start,
                "End_Frame_Image": img_end,
                "Regions": "; ".join([f"{x0}–{x1}" for (x0, x1) in regions]),
            },
        })

    # ---- 軸スケールを推定して物理単位へ換算し、呼吸ごとのCSVを書き出す ----
    rate = float(np.median(sweep_rates)) if len(sweep_rates) >= 3 else SWEEP_RATE_DEFAULT
    estimated = _estimate_scales(records, rate) if scale_enabled else {}
    for rec in records:
        breath_num, df_out = rec["breath"], rec["df"]
        if scale_enabled:
            scale = dict(estimated.get(breath_num, {}))
            # Edi は EdiPeak 値で校正する
            edi_scale = _edi_scale_from_peak(rec["edi_peak"], rec["edi_full"])
            if edi_scale:
                scale["Edi"] = edi_scale
                last_edi_scale = edi_scale         # 有効な目盛りを記憶
            elif last_edi_scale is not None:
                # EdiPeakが*** (NaN) 等で校正不能 → 直前の有効な呼吸のEdi目盛りを流用する
                scale["Edi"] = last_edi_scale
            for nm, sc in scale.items():
                df_out[nm] = _pixels_to_physical(df_out[nm].tolist(), nm, sc)
            if scale:
                breath_scales[breath_num] = scale
        df_out.to_csv(rec["out_path"], index=False)
        print(f"{rec['out_path']} created")

        # サマリー/レポート
        summary_records.append(rec["summary"])
        skip_logs.append({
            "BreathNumber": breath_num, "Status": "OK", "Reason": "", "Details": ""
        })

    # ---- 呼吸ごとスケールを Breaths.csv 用に列展開（P_min,P_max,F_min,...）----
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

    # ---- 診断用: 抽出波形領域の左右端x（X_Start/X_End）と検出カーソル位置を列展開 ----
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

    # ---- レポート出力 ----
    report_df = pd.DataFrame(skip_logs).sort_values("BreathNumber")
    report_df.to_csv(report_path, index=False)

    return len(summary_records), breaths


def re_extract_breath_waveform(img_start_path, img_end_path, end_iframe, scale=None):
    """1呼吸の波形を start/end フレーム画像の差分から再抽出して DataFrame で返す。

    BreathsToCSVs の波形抽出コア（差分領域→終端カーソル右の残像トリム→掃引折り返し領域選択→
    領域結合→末尾の次呼吸トリム→Volume谷除去→末尾30点カット）と同一手順。
    呼吸統合（Fragment等を前後と統合）後に、統合区間から1呼吸ぶんの清浄な波形を作り直す用途。
    既存の {n}.csv どうしを連結すると2呼吸ぶんが積み重なって波形が崩れるため、必ず本関数で再抽出する。

    Parameters
    ----------
    img_start_path : str   統合後呼吸の開始フレーム画像パス（=元の Start_File_Name 相当）
    img_end_path   : str   統合後呼吸の終了フレーム画像パス（=次呼吸 Frame 相当の End_File_Name）
    end_iframe     : int   終了フレームの iframe 番号（終端カーソル検出の直前フレーム導出に使用）
    scale          : dict|None  {name: {"min":, "max":}} 指定時はピクセル→物理単位へ変換。

    Returns
    -------
    pandas.DataFrame | None   列=[X, Pressure, Flow, Volume, Edi]。再抽出不能なら None。
    """
    if not (os.path.exists(img_start_path) and os.path.exists(img_end_path)):
        return None

    # 差分領域抽出
    regions = get_difference_regions(img_end_path, img_start_path)
    if not regions:
        return None

    # 終端カーソル右側の前掃引残像をトリム（直前フレームとの差分で先端を検出）
    before_path = os.path.join(os.path.dirname(img_end_path),
                               f"frame_{int(end_iframe) - 1:08d}.jpg")
    if int(end_iframe) - 1 >= 0 and os.path.exists(before_path):
        cursor_x = _detect_sweep_cursor(img_end_path, before_path)
        regions = _trim_regions_right_of_cursor(regions, cursor_x)

    regions = _select_breath_regions(regions)   # 掃引折り返し時の残像断片を除去

    # 波形抽出
    waveforms = merge_waveforms_across_regions(img_end_path, regions)
    waveforms = _trim_trailing_breath(waveforms)   # 末尾の次呼吸立ち上がりをトリム
    waveforms = _reject_volume_troughs(waveforms)  # Volumeの非生理的V字谷を除去
    length = len(next(iter(waveforms.values()), []))
    if length == 0:
        return None

    # 物理単位へ変換（scale={name:{min,max}}。Breaths.csv の P_min/P_max 等を流用すればOCR不要）
    if scale:
        for nm in ["Pressure", "Flow", "Volume", "Edi"]:
            if nm in scale and nm in waveforms and scale[nm] is not None:
                waveforms[nm] = _pixels_to_physical(waveforms[nm], nm, scale[nm])

    df_out = pd.DataFrame({"X": list(range(1, length + 1))})
    for name in ["Pressure", "Flow", "Volume", "Edi"]:
        df_out[name] = waveforms.get(name, [np.nan] * length)

    if len(df_out) <= 30:   # 末尾30点カット後に消えるなら短すぎ（BreathsToCSVsと同基準）
        return None
    return df_out.iloc[:-30].reset_index(drop=True)


def WaveFromCSV(csv_path):
    df = pd.read_csv(csv_path)
    
    # X列があれば整列する
    if "X" in df.columns:
        df = df.sort_values(by="X").reset_index(drop=True)
        globals()["x_start"] = df["X"].iloc[0]  # グローバル変数に記録
    
    waveforms = {}
    for name in ["Pressure", "Flow", "Volume", "Edi"]:
        if name in df.columns:
            waveforms[name] = df[name].to_numpy()
        else:
            print(f"Warning: {name} 列が見つかりません。NaN で埋めます。")
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


def DetectAbnormalFrames(user_id,df_frames,analyze_id=None):

    """
    与えられたフレーム情報（df_frames）に基づいて、正常基準画像との比較により各フレームが異常かどうかを判定する。

    各フレーム画像について、設定されたROI領域を対象に、
    - 輝度差（平均差）
    - 標準偏差差
    - ヒストグラム相関

    を評価し、いずれかの閾値を超える場合には異常（"1"）と判定する。
    結果は df_frames の "Abnormal" 列に "0" または "1" を追加して返される。

    Parameters:
        user_id (str): 対象ユーザーのID。フレーム画像ディレクトリ取得に使用。
        df_frames (pd.DataFrame): iFrame列とsource_file列を含むDataFrame。各行が1フレームに対応。

    Returns:
        pd.DataFrame: "Abnormal" 列を追加した df_frames。
    
    Raises:
        FileNotFoundError: 正常基準画像 normal.jpg が存在しない場合に発生。

    Notes:
        - フレーム画像は Global_Setting.GS.Dir_Frames(user_id) 配下にあると仮定する。
        - 使用するROIやしきい値は Settings.ROI_LIST_Abnormal_Detection および固定変数で定義される。

    Tests:
        - ✅ Unit Test (abnormal_test_env)
        - ✅ Integration Test (test_detect_abnormal_with_real_images(real_image_test_env))
        - ✅ Validation (validate_roi_difference.py)
    """



    
    # 正常基準画像（比較対象）

    # 実行中スクリプトと同じディレクトリの絶対パスを取得
    script_dir = os.path.dirname(os.path.abspath(__file__))


    base_frame_dir = GS.Dir_Frames(user_id)

    # 正常基準フレームの自動選定:
    # 動画冒頭は黒画面/フェードインの場合があり frame_00000000 を固定で使えない。
    # また異常検知は「同一動画の正常フレーム」を基準にする必要があるため、
    # 対象動画の先頭から走査して最初の「黒でない」フレームを正常基準として採用する。
    # （_load_frame で基準解像度に正規化されるため解像度非依存）
    def _find_reference_frame():
        for _, r in df_frames.iterrows():
            sub = str(r["source_file"]).replace("output_", "").replace(".csv", "")
            cand = os.path.join(base_frame_dir, sub,
                                "frame_" + str(int(r["iFrame"])).zfill(8) + ".jpg")
            im = _load_frame(cand)
            if im is not None and im.mean() > 20:  # 黒画面(mean≈0)を除外
                return im
        return None


    # ROIの座標
    ROI_LIST_Abnormal_Detection = Settings.ROI_LIST_Abnormal_Detection

    # 各種閾値（チューニング可能）
    THRESH_MEAN = 30        # 平均差の閾値（大きいと異常）
    THRESH_STD = 10          # 標準偏差の差（大きいと異常）
    THRESH_HIST_CORR = 0.90  # ヒストグラム相関（小さいと異常）



    # ヒストグラム生成関数
    def compute_hist_corr(gray1, gray2):
        hist1 = cv2.calcHist([gray1], [0], None, [64], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [64], [0, 256])
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    # 正常画像のROI処理
    ref_img = _find_reference_frame()
    if ref_img is None:
        raise FileNotFoundError("正常基準フレームが見つかりません（全フレームが黒/読込不可）")

    ref_rois = []
    ref_grays = []
    ref_stds = []


    for (x1, y1, x2, y2) in Settings.ROI_LIST_Abnormal_Detection:
        roi = ref_img[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ref_rois.append(roi)
        ref_grays.append(gray)
        ref_stds.append(np.std(gray))

    # 異常検出処理
    # imread(JPEGデコード)がI/O律速かつGILを解放するため、フレーム単位でスレッド並列化する。
    # ROIの計算ロジックは直列版と完全に同一（結果一致を確認済み）。
    df = df_frames
    roi_list = Settings.ROI_LIST_Abnormal_Detection

    # フレームパスをベクトル化して構築（iterrowsを回避）
    subfolders = (df["source_file"].astype(str)
                  .str.replace("output_", "", regex=False)
                  .str.replace(".csv", "", regex=False))
    names = ("frame_" + pd.to_numeric(df["iFrame"], errors="coerce")
             .fillna(0).astype(int).astype(str).str.zfill(8) + ".jpg")
    filepaths = (base_frame_dir + os.sep + subfolders + os.sep + names).tolist()

    def _detect_one(filepath):
        img = _load_frame(filepath)
        if img is None:
            print(f"⚠️ 画像読み込み失敗: {filepath}")
            return ["0", 0.0, 0.0, 0.0]

        abnormal = False
        mean_diff = std_diff = hist_corr = 0.0
        for i, (x1, y1, x2, y2) in enumerate(roi_list):
            roi = img[y1:y2, x1:x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            mean_diff = np.mean(cv2.absdiff(ref_grays[i], gray))
            std_diff = abs(np.std(gray) - ref_stds[i])
            hist_corr = compute_hist_corr(ref_grays[i], gray)

            if (mean_diff > THRESH_MEAN or
                std_diff > THRESH_STD or
                hist_corr < THRESH_HIST_CORR):
                abnormal = True
                break  # どちらかで異常なら早期終了

        return ["1" if abnormal else "0", mean_diff, std_diff, hist_corr]

    # ThreadPoolExecutor.map は入力順を保持するため df 行と整列する
    from concurrent.futures import ThreadPoolExecutor
    total = len(filepaths)
    results = []
    with ThreadPoolExecutor(max_workers=ABNORMAL_WORKERS) as executor:
        for i, r in enumerate(tqdm(executor.map(_detect_one, filepaths),
                                   total=total, desc="DetectAbnormalFrames"), 1):
            results.append(r)
            if i % 200 == 0 or i == total:
                _report_progress(user_id, analyze_id,
                                 message=f"Detecting abnormal frames ({i}/{total})",
                                 current=i, total=total)

    # 結果を保存（df.indexに整列）
    df[["Abnormal", "MeanDiff", "StdDiff", "HistCorr"]] = pd.DataFrame(results, index=df.index)

    return df