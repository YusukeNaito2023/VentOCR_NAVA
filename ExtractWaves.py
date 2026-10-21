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
    解像度非依存化のため、ROI/Y_RANGES定義の基準解像度に全フレームを揃える。
    v1.4: アプリがフレームの画像をその場で作る（manifest の frame_images: on_demand）ので、必ず cv2.imread にパスを渡して読む。
    BreathsToCSVs の間は同じ画像（終了画像は3回読む）を読み直さないよう、直近の数枚を覚えておく。"""
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


START_BACK_SEC = 0.6   # 開始画像は呼吸の区切りのこの秒数前（v1.3 までは 6 フレーム決め打ち）
FRAME_CACHE_SIZE = 8   # 1呼吸で読む画像は開始・終了とそれぞれの直前の4枚
_frame_cache = None    # BreathsToCSVs の間だけ OrderedDict（path -> 基準解像度の画像）

# --- 欠損値補完とスムージングを行うかどうか ---

TRIM_TO_BREATH = True  # v1.3: 波形を呼吸の範囲（吸気の始まり〜次の吸気の始まり）に切り詰める
SMOOTHING = False   # v1.3: 平滑化が実際に効くようになったので既定を OFF に（engine_options.json も OFF）
IMPUTATION = True
# v1.4: abnormal_workers（DetectAbnormalFrames の並列読み込み）は廃止。古い設定に残っていても無視する

def set_config(settings):
    global SMOOTHING, IMPUTATION, TRIM_TO_BREATH
    SMOOTHING = settings.get("post_ocr", {}).get("smoothing", False)
    IMPUTATION = settings.get("post_ocr", {}).get("imputation", True)
    TRIM_TO_BREATH = settings.get("post_ocr", {}).get("trim_to_breath", True)

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

# --- 列の範囲から波形の点を決めるためのパラメータ（v1.3） ---
# v1.3 変更点:
#   v1.2 は列ごとに「色相が最も近い1画素」を採用していた。急峻な箇所（吸気→呼気の切り替わりなど）では
#   1列に 30〜60px の縦線があり（縮小・JPEG のにじみで数列に広がる）、その中のどの高さが選ばれるかが
#   ほぼ偶然で決まるため、隣の列と上下に跳ねて画面に無いスパイクになっていた（例: サンプル B10 の t≈0.73s）。
#   v1.3 は列ごとに「線が通った範囲」を取り、
#     1) 縦に続いた画素のかたまりの中から、隣の列とつながる線を呼吸全体で最適に選ぶ（ノイズや文字を除く）
#     2) 範囲の中でなめらかな経路を呼吸全体で最適に引く（縦線は坂になり、跳ねない）
#     3) 範囲が経路から局所的に突き出している所（V字の谷・山の先端）はその値に固定して引き直す
#   ことで、スパイクを出さずに先端（呼気の最大流量、圧の行き過ぎ）を残す。
RUN_JOIN = 4            # 同じ列で、これ未満の画素の切れ目はつないで1つのかたまりとする
SEG_GAP = 12            # 列番号の差がこれを超えたら別の線とみなして経路を切る
RUN_HUE_WEIGHT = 0.5    # かたまりの選択で、色相のずれに掛ける重み（主はつながり）
PATH_STEP = 0.5         # 経路の高さの刻み（px）
TIP_MARGIN = 2.0        # 範囲が経路から これ以上 突き出している列を先端の候補とする（px）
TIP_DROP = 2.0          # 先端の両側で、範囲の端が これ以上 引っ込んでいること（＝坂の途中ではない）
TIP_WIN = 8             # 先端の両側を調べる列数
TIP_SUPPORT = 2         # 先端の深さ（±TIP_SUPPORT_PX）に届いている列がこれ以上あること（1列だけの文字・ノイズを除く）
TIP_SUPPORT_PX = 1.5


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
# Flow の積分から求めた目盛りは実際より約 5% 小さく出る。
# v1.3: 3本の動画（261呼吸）で画面の目盛りと照合して 1.06 → 1.05（急な区間を除いた推定で最も一致する値）
FLOW_GAIN = 1.05
FLOW_WINDOW = 6             # Volume の増減と Flow の積分を比べる区間の長さ（px）
# v1.3: Flow が隣の列と これを超えて変わる区間（V字の先端・揺れ）は当てはめに使わない（px）。
# 先端を残すようになって Volume の増減と合わなくなり、推定がばらついたため
FLOW_STEEP_PX = 3


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
    0 付近と上端に張り付いた Volume（呼気の終わり、目盛りからはみ出した吸気）と、
    Flow が急に変わる区間（FLOW_STEEP_PX）は使わない。"""
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
def _column_runs(img, waveform_name):
    """列ごとに、波形色の画素の縦のかたまり [(lo, hi, hue_dist)] を返す（lo/hi はバンド下端からの高さ）。
    画素の判定（色相・彩度・明度・左端の文字除外・白いペン先）は v1.2 と同じ。"""
    h, w, _ = img.shape
    target_hue = TARGET_HUES.get(waveform_name)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cols = []
    for x in range(w):
        column = img[:, x, :]
        hsv_col = hsv_img[:, x, :]
        H, S, V = hsv_col[:, 0], hsv_col[:, 1], hsv_col[:, 2]
        global_x = x + globals().get("x_start", 0)

        # 判定: Edi はスイープ位置(847)以降を無効化（v1.1 と同一仕様）
        if waveform_name == "Edi" and 847 <= global_x:
            cols.append([])
            continue

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

        if not valid.any():
            # 補助判定: 白色（スイープカーソル先端）で再挑戦
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
    """各列のかたまりから、隣の列とつながる1本の線を選ぶ（Viterbi）。
    コスト = Σ(隣のかたまりとの隙間^2 / 列の間隔) + RUN_HUE_WEIGHT × 色相のずれ。
    列番号の差が SEG_GAP を超える所で区間を分ける。戻り値: (列ごとの (lo, hi) or None, 区間ごとの列番号リスト)"""
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
    """各列の範囲 [lo_i, hi_i] の中で Σ(隣の列との差^2 / 列の間隔) が最小の経路（PATH_STEP 刻みの Viterbi）。"""
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
    """列 [i, j] の先端 ext から左右へ TIP_WIN 列以内に TIP_DROP 以上引っ込む列があり、
    それより手前に ext を超えて突き出す列が無いこと（＝局所的な山・谷の先端。坂の途中ではない）。"""
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
    """範囲（中心線に直した下端 a・上端 b）が経路 y から局所的に突き出している先端 {位置: 値} を返す。"""
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


# imageを渡すと波形の時系列データを返してくれる内部関数
def _extract_waveform(img, target_bgr, max_dist=40, waveform_name=None, y_offset=0):
    """列ごとの「線が通った範囲」から波形の時系列（バンド下端からの高さ, 欠損は NaN）を返す（v1.3）。"""
    cols = _column_runs(img, waveform_name)
    out = np.full(len(cols), np.nan)
    chosen, segs = _choose_runs(cols)
    if not segs:
        return list(out)
    # 線の太さ（緩やかな所のかたまりの高さ）。範囲の上下端を線の中心へ寄せるのに使う
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
            # v1.2 までは移動平均の結果を直後に上書きしており、平滑化は効いていなかった（v1.3 で修正）。
            # 既定は OFF。ON にすると V字の先端が削れる
            series = series.rolling(window=10, center=True, min_periods=1).mean()
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


# --- 呼吸の範囲への切り詰め ---
# v1.3 追加:
#   呼吸の区切り（Cdyn の表示の更新）は吸気の始まりと微妙にずれるため、開始画像は区切りの 0.6 秒前
#   （Frame-6）から取り、終了画像は次の呼吸の区切りまで取っている。そのため波形 CSV の先頭に前の呼吸の
#   呼気の終わり（約 0.4〜0.6 秒）、末尾に次の呼吸の吸気の立ち上がりが入っていた。
#   Flow の吸気の立ち上がりを見つけ、先頭は最初の吸気の始まりまで、末尾は「呼気を伴わない最後の吸気」
#   （＝次の呼吸の途中で切れた吸気）の始まりまで削る。さらに、次の呼吸で見つけた吸気の始まりの時刻で
#   前の呼吸の末尾を切る（短すぎて自分では吸気と判定できない立ち上がりも除くため）。
#   削った分は Breaths.csv の Wave_T0（最初の点の時刻・秒）/ Wave_Rate（px/秒）で時刻を保つ。
#   計算は呼吸ごとに Flow の1次元の走査だけ。
ONSET_PX = 3        # Flow が基線からこれを超えたら吸気の流れ（px）
ONSET_RUN = 5       # 吸気の流れがこれ以上の列続くこと
ONSET_PEAK = 8      # その吸気が基線からこれ以上に届くこと（px）
QUIET_RUN = 5       # 吸気の前に、吸気でない列がこれ以上あること（先頭に入った前の呼吸の吸気の残りを除く）
ONSET_FRAC = 0.2    # 吸気の最大値のこの割合を超えた点から、立ち上がりの足元までさかのぼる
EXP_PX = 3          # 呼気の流れ（基線よりこれ以上下, px）
TRIM_LEAD = 3       # 吸気の始まりの何列前から残すか


def _inspiration_onsets(flow):
    """flow: Flow（基線=0、上が吸気、px、NaN 可）。吸気の始まり（立ち上がりの足元）の列番号の一覧。"""
    raw = np.asarray(flow, dtype=float)
    g = pd.Series(raw).interpolate(limit_area="inside").to_numpy()   # 欠けた列は判定のときだけつなぐ
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
            # 立ち上がりが欠けた列（掃引の折り返しの継ぎ目など）をまたぐとき、欠ける直前が呼気なら
            # つないだ線が作った偽の立ち上がりなので吸気の始まりとみなさない
            k = c + int(np.argmax(np.isnan(raw[c:]))) if np.isnan(raw[c:i + 1]).any() else None
            before = raw[:k][np.isfinite(raw[:k])] if k is not None else None
            if before is None or len(before) == 0 or before[-1] >= -EXP_PX:
                out.append(c)
        i = j + 1
    return out


def _breath_trim_range(flow, valid_rows):
    """残す行の範囲 (head, tail) と最初の吸気の始まり（無ければ None）を返す。
    valid_rows: いずれかの波形に値がある行（先頭・末尾の空の行も削る）。"""
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
    # 最後の吸気の後に呼気が無い＝次の呼吸の吸気が途中で切れて入っている（2呼吸ぶん入った呼吸の2つ目は残す）
    if last != onsets[0] and not (np.nan_to_num(np.asarray(flow[last:], float)) < -EXP_PX).any():
        tail = max(head + 1, min(hi, last - TRIM_LEAD))
    return head, tail, onsets[0]


def _first_sample_time(start_frame, x0, cursor_start, rate, offset):
    """波形の最初の点の時刻（秒）。アプリの ResultPackage.sample_times と同じ求め方:
    開始画像のペン先（掃引位置 − 消去帯の幅 offset）から逆算する。折り返しを含む呼吸は開始画像の時刻。"""
    ts = float(start_frame) / GS.FPS
    if cursor_start is not None and x0 is not None and 0 < cursor_start \
            and 0 <= cursor_start - x0 <= max(200, 3 * offset):
        return ts - (cursor_start - offset - x0) / rate
    return ts


def _trim_records_to_breaths(records, rate):
    """records（呼吸の時間順）を切り詰める。rec に head / tail / wave_t0 を書き込む。"""
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
        # 次の呼吸の吸気の始まりより後ろは、前の呼吸の末尾から削る
        for a, b in zip(records, records[1:]):
            # 自分の吸気の始まりが無い呼吸（壊れた呼吸）は時刻も怪しいので切らない
            if a["onset_time"] is None or b["onset_time"] is None or b["onset_time"] <= a["onset_time"]:
                continue
            cut = int(np.ceil((b["onset_time"] - a["t_first"]) * rate)) - TRIM_LEAD
            if a["head"] + 1 < cut < a["tail"]:
                a["tail"] = cut
    for r in records:
        r["wave_t0"] = r["t_first"] + r["head"] / rate


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
        if not (os.path.isfile(img_start) and os.path.isfile(img_end)):
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

def BreathsToCSVs(*args, **kwargs):
    """_BreathsToCSVs を、フレームの画像を覚えておく（_frame_cache）状態で呼ぶ。"""
    global _frame_cache
    from collections import OrderedDict
    _frame_cache = OrderedDict()
    try:
        return _BreathsToCSVs(*args, **kwargs)
    finally:
        _frame_cache = None


def _BreathsToCSVs(breaths, frames, frames_root, output_dir, user_id, overwrite=False, report_path=None, analyze_id=None, reader=None, settings=None):
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
    # v1.4: Abnormal は initial_ocr が付ける（列が無ければ異常なしとみなす）
    frames_idx_abn = (pd.to_numeric(frames["Abnormal"], errors="coerce").fillna(0).astype(int)
                      if "Abnormal" in frames else pd.Series(0, index=frames.index))
    frames_idx_abn.index = frames["Frame"]
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
    start_back = max(1, round(START_BACK_SEC * GS.FPS))   # v1.4: 10 fps で v1.3 と同じ 6 フレーム
    adjusted_frame_start = (base_frame - start_back).fillna(frame_min).astype(int).clip(frame_min, frame_max)
    breaths["_Start_Frame"] = adjusted_frame_start   # v1.3: 波形の時刻（Wave_T0）を求めるため。最後に消す


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
    breaths = breaths.iloc[:-1]   # 0.1.1: 呼吸が無いときに drop(index[-1]) で落ちないように

    adjusted_frame_end = pd.to_numeric(breaths["End_Frame"], errors="coerce").fillna(frame_max).astype(int)
    adjusted_frame_end = adjusted_frame_end.clip(frame_min, frame_max)

    breaths["_End_Frame"] = adjusted_frame_end   # v1.4: 終了画像が Abnormal かを見るため。最後に消す
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

    # v1.5: 使う画像（開始・終了とそれぞれの直前）を先に、区間・フレーム番号の小さい順に開いておく。
    # アプリ（engine_api 3）は開かれた画像を区間動画から作り、前へ進む読み方なら位置を指定し直さずに読み進められる。
    # 下の処理は呼吸ごとに終了 → 開始の順（後ろへ戻る）に開くので、v1.4 はそのたびに区間動画のシークが起きていた
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

        # v1.4: 波形を読む開始・終了画像が Abnormal（ポップアップなど）なら書き出さない。
        # 呼吸の途中の Abnormal は問わない（終了画像から波形が読めればよい。Hamilton と同じ考え方）。
        # v1.3 までは呼吸の中に1枚でも Abnormal があれば書き出さなかった
        abn_start = int(frames_idx_abn.get(int(row["_Start_Frame"]), 0))
        abn_end = int(frames_idx_abn.get(int(row["_End_Frame"]), 0))
        if abn_start or abn_end:
            skip_logs.append({
                "BreathNumber": breath_num, "Status": "SKIPPED",
                "Reason": "abnormal frames", "Details": f"start={abn_start}, end={abn_end}",
            })
            continue

        # 画像存在チェック
        img_start = str(row["Start_File_Name"])
        img_end   = str(row["End_File_Name"])
        if not (os.path.isfile(img_start) and os.path.isfile(img_end)):
            skip_logs.append({
                "BreathNumber": breath_num, "Status": "SKIPPED",
                "Reason": "image not found",
                "Details": f"start={os.path.isfile(img_start)}, end={os.path.isfile(img_end)}",
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
        if end_iframe - 1 >= 0 and os.path.isfile(before_path):
            cursor_x = _detect_sweep_cursor(img_end, before_path)
            regions = _trim_regions_right_of_cursor(regions, cursor_x)

        regions = _select_breath_regions(regions)   # 掃引折り返し時の残像断片を除去

        # 開始フレーム側の掃引カーソルx（開始フレームと直前フレームの差分で検出）
        cursor_x_start = None
        before_start = os.path.join(os.path.dirname(img_start),
                                    f"frame_{start_iframe - 1:08d}.jpg")
        if start_iframe - 1 >= 0 and os.path.isfile(before_start):
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

    # ---- 軸スケールを推定して物理単位へ換算し、呼吸ごとのCSVを書き出す ----
    rate = float(np.median(sweep_rates)) if len(sweep_rates) >= 3 else SWEEP_RATE_DEFAULT
    estimated = _estimate_scales(records, rate) if scale_enabled else {}
    _trim_records_to_breaths(records, rate)   # 目盛りは切り詰める前の波形で推定する
    breath_wave_time = {}
    for rec in records:
        breath_num = rec["breath"]
        df_out = rec["df"].iloc[rec["head"]:rec["tail"]].copy()   # X は元の列番号のまま
        breath_wave_time[breath_num] = (rec["wave_t0"], rate, rec["head"], len(rec["df"]) - rec["tail"])
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

    # ---- v1.3: 波形の最初の点の時刻と掃引の速さ（アプリはこれで各点の時刻を求める）、削った行数 ----
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

    # ---- レポート出力 ----
    # 0.1.1: 波形を書き出す候補の呼吸が無い（呼吸の区切りが2つ以下。合わない動画など）と skip_logs が空になり、
    # 列の無い表を並べ替えて KeyError で解析が止まっていた。空でも列を持つ表にする
    report_df = pd.DataFrame(skip_logs, columns=["BreathNumber", "Status", "Reason", "Details"]).sort_values("BreathNumber")
    report_df.to_csv(report_path, index=False)

    return len(summary_records), breaths


def re_extract_breath_waveform(img_start_path, img_end_path, end_iframe, scale=None):
    """1呼吸の波形を start/end フレーム画像の差分から再抽出して DataFrame で返す。

    BreathsToCSVs の波形抽出コア（差分領域→終端カーソル右の残像トリム→掃引折り返し領域選択→
    領域結合→末尾の次呼吸トリム→Volume谷除去→末尾30点カット）と同一手順。
    呼吸統合（Fragment等を前後と統合）後に、統合区間から1呼吸ぶんの清浄な波形を作り直す用途。
    既存の {n}.csv どうしを連結すると2呼吸ぶんが積み重なって波形が崩れるため、必ず本関数で再抽出する。
    v1.3: 呼吸の範囲への切り詰め（_trim_records_to_breaths）は時刻の情報が要るため、ここでは行わない。

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
    if not (os.path.isfile(img_start_path) and os.path.isfile(img_end_path)):
        return None

    # 差分領域抽出
    regions = get_difference_regions(img_end_path, img_start_path)
    if not regions:
        return None

    # 終端カーソル右側の前掃引残像をトリム（直前フレームとの差分で先端を検出）
    before_path = os.path.join(os.path.dirname(img_end_path),
                               f"frame_{int(end_iframe) - 1:08d}.jpg")
    if int(end_iframe) - 1 >= 0 and os.path.isfile(before_path):
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


class AbnormalFrameDetector:
    """v1.4: 異常フレーム（ポップアップなど）の判定を initial_ocr の中で、メモリにあるフレームで行う。

    v1.3 までは後処理の DetectAbnormalFrames が全フレームの JPEG を読み直していた（1時間の動画で 36,000 枚）。
    判定の中身（ROI_LIST_Abnormal_Detection の範囲を基準と比べ、平均差・標準偏差の差・ヒストグラム相関の
    どれかが閾値を超えたら異常）と閾値は v1.3 と同じ。

    基準: v1.3 は動画の先頭の「黒でない」フレーム1枚だった。initial_ocr は区間（既定 60 秒）ごとに動くので、
    区間の黒でないフレームの画素ごとの中央値にする（区間の先頭がポップアップでも基準が崩れないように）。

    使い方: add(フレーム) を initial_ocr の行と同じ順に呼び、最後に results() で
    [(Abnormal, MeanDiff, StdDiff, HistCorr), ...] を受け取る。
    """

    THRESH_MEAN = 30         # 平均差の閾値（大きいと異常）
    THRESH_STD = 10          # 標準偏差の差（大きいと異常）
    THRESH_HIST_CORR = 0.90  # ヒストグラム相関（小さいと異常）
    BLACK_MEAN = 20          # 画面全体の平均がこれ以下のフレームは黒画面（基準に使わない）
    MAX_REF_FRAMES = 200     # 基準の中央値に使うフレームの数の上限（等間隔に間引く）

    def __init__(self, roi_list=None):
        self.roi_list = list(roi_list or Settings.ROI_LIST_Abnormal_Detection)
        self.strips = []     # フレームごと: [ROIごとのグレースケール画像]
        self.black = []      # フレームごと: 黒画面か

    def add(self, bgr):
        """基準解像度（Settings.BASE_RESOLUTION）に揃えたカラーフレームを1枚加える。"""
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
            # 区間がすべて黒画面（または空）: 波形は読めないので異常にする
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
                    break  # どちらかで異常なら早期終了
            out.append((abnormal, mean_diff, std_diff, hist_corr))
        return out
