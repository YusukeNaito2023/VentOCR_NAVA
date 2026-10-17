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
from pathlib import Path
import threading

ENGINE_VERSION = "1.1.0"
VENTOCR_TESTED = "0.5.5"


_EASYOCR_LOCKFILE = "easyocr_init.lock"


def _get_easyocr_model_dir():
    # Engineごとではなく、ユーザー共通の永続保存先にする
    model_dir = os.path.join(".", "_runtime_models", "easyocr")
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def _easyocr_model_ready(model_dir: str) -> bool:
    p = Path(model_dir)
    if not p.exists():
        return False

    # EasyOCRの検出/認識モデルが1つでも入っていれば最低限OKとみなす
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


def _ensure_easyocr_reader_ready(gpu_enabled: bool):
    model_dir = _get_easyocr_model_dir()

    # まず完成済みなら即使用
    if _easyocr_model_ready(model_dir):
        print(f"[EasyOCR] using existing model: {model_dir}")
        return Reader(
            ["en"],
            gpu=gpu_enabled,
            model_storage_directory=model_dir,
            download_enabled=True
        )

    # 未準備ならプロセス間ロック
    fd, lock_path = _acquire_easyocr_file_lock(model_dir)
    try:
        # 待っている間に他プロセスが終わらせた可能性があるので再確認
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

def initial_ocr(frames, total_frames, subfolder, settings={}, progress_callback=None, frames_dir=None):
    """
    v1.1: カラーフレームジェネレータ対応。
    - gc.collect()をループ内から除去
    - ROIバッチOCR
    - カラーフレームをジェネレータで受け取り、内部でグレースケール変換

    Args:
        frames: カラーフレームのジェネレータ (BGR, numpy配列)。list化禁止。
        frames_dir: str (未使用、インターフェース互換のため)
    """
    from easyocr import Reader

    sampling_size = settings["initial_ocr"]["sampling_size"]
    skip_keys = settings["initial_ocr"]["skip_keys"]
    similarity_thresh = settings["initial_ocr"]["similarity_thresh"]
    gpu_enabled = settings["initial_ocr"]["use_gpu"]

    # 差分ゲーティング設定（未設定の古いengine_optionsでも動くようdefault付き）
    _io = settings["initial_ocr"]
    diff_gating = _io.get("diff_gating", True)
    diff_thresh = _io.get("diff_thresh", 2.0)
    diff_thresh_cdyn = _io.get("diff_thresh_cdyn", 0.5)  # Cdynは安全側で低閾値
    # recognizeのconfがこの閾値未満ならreadtext(検出あり)にフォールバック
    recognize_conf_fallback = _io.get("recognize_conf_fallback", 0.5)

    reader = _ensure_easyocr_reader_ready(gpu_enabled)
    print(f"GPU:{gpu_enabled}")

    # ---- Batch OCR configuration ----
    OCR_BATCH_SIZE = 16

    # 検出スキップ(recognize)対象: 単一数値ROI。
    # IE_Ratio/Triggeredは複数トークン/検出が有効なのでreadtextのまま残す。
    RECOGNIZE_KEYS = {"VTe", "EdiPeak", "Cdyn", "Ppeak", "VTi"}
    # 差分ゲーティング対象: 毎フレームOCRしている数値表示ROI
    GATED_KEYS = {"VTe", "EdiPeak", "Cdyn", "IE_Ratio", "Ppeak", "VTi"}
    # 毎フレームOCRするROI（それ以外は sampling_size ごと）。値は呼吸ごとにしか変わらないので差分ゲーティングで大半を省ける
    EVERY_FRAME_KEYS = {"Cdyn", "IE_Ratio", "VTe", "EdiPeak", "Ppeak", "VTi"}

    Data = []
    frame_count = 0

    # ヒストグラム比較用画像読み込み
    x, y, w, h = Settings.ROIs["Triggered"]
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(BASE_DIR, "sample_frame.jpg")
    background_img = cv2.imread(image_path)
    if background_img is None:
        raise FileNotFoundError(f"背景画像が見つかりません: {image_path}")
    gray_background_image = cv2.cvtColor(background_img, cv2.COLOR_BGR2GRAY)
    background_hist = cv2.calcHist([gray_background_image[y:y+h, x:x+w]], [0], None, [256], [0, 256])
    cv2.normalize(background_hist, background_hist, 0, 1, cv2.NORM_MINMAX)

    # Pre-compute ROI keys and positions once
    roi_items = list(Settings.ROIs.items())
    all_keys = [k for k, _ in roi_items]
    extra_keys = [f"{k}_raw" for k in all_keys] + [f"{k}_conf" for k in all_keys]
    column_order = all_keys + extra_keys

    # ---- Batch accumulator ----
    # Each entry: (frame_idx, key, roi_image)
    batch_rois = []
    batch_meta = []  # (frame_idx_in_pending, key, preprocess_type)
    # pending_frames[i] = extracted_values dict for frame i
    pending_frames = []
    pending_start_frame = 0

    # 差分ゲーティング用の状態（ROIキーごと）
    prev_roi = {}        # key -> 最後にOCRした時のROI(ndarray)
    last_ocr_idx = {}    # key -> そのROIをOCRしたpending_frames index
    reuse_links = []     # (target_frame_rel, key, source_frame_rel)

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
                # 検出(CRAFT)をスキップし認識(CRNN)のみ実行。ROI全体を1行として読む
                rc = reader.recognize(roi_img)
                conf_rc = max((c for _, _, c in rc), default=0.0)
                # 低信頼時はreadtext(検出あり)にフォールバックし信頼度の高い方を採用。
                # recognizeはタイトな数字ROIで稀に端のノイズを桁と誤認する（その時conf低）ため。
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
                ev[key] = value
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

        cdyn_folder = f"./temp/Cdyn_ROIs_{subfolder}"
        os.makedirs(cdyn_folder, exist_ok=True)

        # カラーフレームジェネレータから順次受け取り、グレースケール変換
        for color_frame in frames:
            # 解像度非依存化: ROI定義の基準解像度(Settings.BASE_RESOLUTION)に正規化する
            if (color_frame.shape[1], color_frame.shape[0]) != Settings.BASE_RESOLUTION:
                interp = cv2.INTER_AREA if color_frame.shape[1] > Settings.BASE_RESOLUTION[0] else cv2.INTER_LINEAR
                color_frame = cv2.resize(color_frame, Settings.BASE_RESOLUTION, interpolation=interp)
            image = np.dot(color_frame[..., :3], [0.114, 0.587, 0.299]).astype(np.uint8)
            if image is None:
                print(f"Error: Could not read image at frame {frame_count}, skipping...")
                frame_count += 1
                pbar.update(1)
                continue

            frame_rel = len(pending_frames)
            extracted_values = {"_frame": frame_count}
            pending_frames.append(extracted_values)

            for key, (x, y, w, h) in roi_items:
                if key in skip_keys:
                    extracted_values[key] = None
                    extracted_values[f"{key}_raw"] = ""
                    extracted_values[f"{key}_conf"] = "0.00"
                    continue

                if frame_count % sampling_size == 0 or key in EVERY_FRAME_KEYS:
                    roi = image[y:y+h, x:x+w]

                    if key == "EdiPeak":
                        roi = _preprocess_for_edi(roi)

                    # 差分ゲーティング: 前回OCR時のROIとほぼ同じならOCRを省略し、
                    # 前回OCRしたフレームの結果をfinalize時にコピーする
                    if diff_gating and key in GATED_KEYS and key in prev_roi \
                            and prev_roi[key].shape == roi.shape:
                        thresh = diff_thresh_cdyn if key == "Cdyn" else diff_thresh
                        mad = np.abs(roi.astype(np.int16) - prev_roi[key].astype(np.int16)).mean()
                        if mad < thresh:
                            reuse_links.append((frame_rel, key, last_ocr_idx[key]))
                            continue

                    batch_rois.append(roi)
                    batch_meta.append((frame_rel, key, key))
                    if key in GATED_KEYS:
                        prev_roi[key] = roi.copy()
                        last_ocr_idx[key] = frame_rel
                else:
                    extracted_values[key] = None
                    extracted_values[f"{key}_raw"] = ""
                    extracted_values[f"{key}_conf"] = "0.00"

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

        # 差分ゲーティングで再利用したフレームに、参照元フレームのOCR結果を反映。
        # pending_framesはセグメント終端まで保持されるため、参照元の値は確定済み。
        for tgt, key, src in reuse_links:
            ev_t = pending_frames[tgt]
            ev_s = pending_frames[src]
            ev_t[key] = ev_s.get(key)
            ev_t[f"{key}_raw"] = ev_s.get(f"{key}_raw", "")
            ev_t[f"{key}_conf"] = ev_s.get(f"{key}_conf", "0.00")

        _finalize_pending()

    columns = ["Frame"] + column_order
    df = pd.DataFrame(Data, columns=columns)
    return df

        
        
def post_ocr(df, Analyze_Id, user_ID, settings={}) :

    # post_ocr 内の各サブ段階を overall 進捗へ流すヘルパー（停止しているように見える問題対策）。
    # 既存の ExtractWaves._report_progress（progress_io ベース・ベストエフォート）を流用する。
    def _pp(msg, cur=0, tot=0):
        try:
            ew._report_progress(user_ID, Analyze_Id, message=msg, current=cur, total=tot)
        except Exception:
            pass

    _pp(f"post_ocr: 呼吸番号を付与中 ({len(df)} rows)")
    df["IE_Ratio"] = df["IE_Ratio_raw"].apply(parse_IE_ratio)


    df = CreateBreathNumber(df) # Cdynの値を前後の値などで修正し、その後Breath Numberを作成

        
    _pp("post_ocr: 時刻・呼吸代表値を集計中")
    # 動画でのおおよその時間をFPSをもとにDFに書き込み
    frame_interval = 1 / GS.FPS
    df["Time"] = pd.to_timedelta(df["Frame"] * frame_interval, unit="s")
    df["Time"] = df["Time"].apply(lambda x: f"{x.seconds//3600}:{(x.seconds%3600)//60:02}:{x.seconds%60:02}:{int(x.microseconds/10000):02}")

   
   # 次の呼吸のVTi, VTeを取得（VTiなどは途中で変わるので一番初めの値を取得。これが実際の換気量を反映している）
    df['Real_VTe'] = Shift_Values(df, "VTe")
    # Ppeak / VTi も呼吸の終わり（=次の呼吸の先頭）で更新されるので、次の呼吸の先頭の値がその呼吸の値。
    # 波形の軸スケール推定（ExtractWaves._estimate_scales）に使う
    for key in ("Ppeak", "VTi"):
        if key in df.columns:
            df[key] = pd.to_numeric(df[key], errors="coerce")
            df[f"Real_{key}"] = Shift_Values(df, key)

    # 1呼吸ごとのデータフレームも作成
    df_breath = df.loc[df.groupby('BreathNumber')['Frame'].idxmin()]


    # Edi Peakの補完
    # 各BreathNumberごとに「最後に出てきた非NaNのEdiPeak」を取得
    last_peak_per_breath = (
        df.groupby("BreathNumber")["EdiPeak"]
        .apply(lambda s: s.dropna().iloc[-1] if not s.dropna().empty else np.nan)
    )

    # 代表値で全行を更新
    df_breath["EdiPeak"] = df_breath["BreathNumber"].map(last_peak_per_breath)


    # Vteの補完
    # 各BreathNumberごとに「最後に出てきた非NaNのVTe」を取得
    last_peak_per_breath = (
        df.groupby("BreathNumber")["VTe"]
        .apply(lambda s: s.dropna().iloc[-1] if not s.dropna().empty else np.nan)
    )

    # 代表値で全行を更新
    df_breath["VTe"] = df_breath["BreathNumber"].map(last_peak_per_breath)






    ###########################################################################################
    # １呼吸ごとの波形解析
    base_dir = os.path.join(".", user_ID)

    # df_framesにAbnormal列をつけて返す
    df_frames = df
    ew.set_config(settings)
    _pp(f"post_ocr: 異常フレーム検知中 ({len(df_frames)} frames)")
    print(f"[post_ocr] DetectAbnormalFrames starting ({len(df_frames)} frames)...")
    df_frames = ew.DetectAbnormalFrames(user_ID,df_frames,analyze_id=Analyze_Id)
    print(f"[post_ocr] DetectAbnormalFrames done")

    df_breath.to_csv("breath.csv", index=False)
    df_frames.to_csv("frames.csv")
    df_breath.to_csv("breath.csv")

    # Abnormalがついたものは除外しそれ以外は個別のCSVに書き出し
    _pp(f"post_ocr: 呼吸ごとのCSVを書き出し中 ({len(df_breath)} breaths)")
    print(f"[post_ocr] BreathsToCSVs starting ({len(df_breath)} breaths)...")
    # 軸スケールは Real_Ppeak / Real_VTi と波形から推定する（軸の数字はOCRしないので reader は不要）
    Effective_Breaths, df_breath = ew.BreathsToCSVs(df_breath,df_frames,GS.Dir_Frames(user_ID),GS.Dir_BreathCSVs(user_ID), user_ID, analyze_id=Analyze_Id, settings=settings)
    print(f"[post_ocr] BreathsToCSVs done ({Effective_Breaths} effective breaths)")

    # Detect Sweep Line（既定OFF）: 有効時は各フレームに掃引カーソルを描き込んで上書き保存する（診断用）
    if settings.get("post_ocr", {}).get("detect_sweep_line", False):
        print("[post_ocr] Detect Sweep Line: annotating frames with sweep cursor...")
        _n_sweep = ew.annotate_sweep_lines(df_frames, GS.Dir_Frames(user_ID))
        print(f"[post_ocr] Detect Sweep Line: annotated {_n_sweep} frames")
    

 

    # fields.txt を作成（セクション形式）
    # 保存先は Main.py のアーカイブ処理と同じ Archive_Dir_For を使う。
    # 別保存先ディスク(VentOCR_Archive_*)選択時に fields.txt/scale_meta.json が
    # 既定(Temporary)側へ取り残されて Label がグラフを描けない不具合を防ぐ。
    archive_dir = GS.Archive_Dir_For(user_ID, Analyze_Id)
    os.makedirs(archive_dir, exist_ok=True)

    fields_path = os.path.join(archive_dir, "fields.txt")
    with open(fields_path, "w", encoding="utf-8") as f:
        f.write("[waves]\n")
        for w in ["Pressure", "Flow", "Volume", "Edi"]:
            f.write(f"{w}\n")
        f.write("\n")
        f.write("[fields]\n")
        for field in ["Cdyn", "VTe", "EdiPeak", "Real_VTe", "Real_VTe2", "Real_Ppeak", "Real_VTi", "Alert"]:
            f.write(f"{field}\n")

    # scale_meta.json を保存（Engine特有のスケール調整設定: ステップ幅/単位/対称性）。
    # Label(Flask/JS) がArchiveから読み、+/- ボタンのステップ等をEngineごとに適用する。
    try:
        import json as _json
        with open(os.path.join(archive_dir, "scale_meta.json"), "w", encoding="utf-8") as f:
            _json.dump(ew.scale_meta(), f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[post_ocr] scale_meta.json 書き出し失敗: {e}")

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


def read_text(reader, roi):
    """Legacy wrapper: calls readtext then parses. Kept for compatibility."""
    result = reader.readtext(roi)
    return _parse_ocr_result(result)


def normalize_ie_ratio(raw_texts: list[str], numeric_parts: list[str]) -> str:
    """
    OCR生文字から IE 表記を厳密正規化する。
      - 出力は必ず '1:X' または 'X:1'
      - X は 1 以上、かつ小数点以下 1 桁（四捨五入）
    想定誤読:
      '1 :.21'→'1:2.1', '1.21'→'1:2.1', '1.2'→'1:2.0', '1.2.0'→'1:2.0',
      '01:02.1'→'1:2.1', 'T:17'→'1:1.7', 'E.17'→'1:1.7', '1, 2.2'→'1:2.2'
    """
    import re

    # 1) テキスト正規化
    s = " ".join(raw_texts)
    s = s.upper()
    s = (s.replace("：", ":")
           .replace(",", ":")
           .replace("I", "1").replace("L", "1").replace("T", "1"))
    s = re.sub(r"\bE[.:]?", "1:", s)      # 例: 'E.17' → '1:17'
    s = re.sub(r"\s+", "", s)

    # 2) 片側トークンを数値化する補助
    def side_to_float(tok: str) -> float | None:
        tok = re.sub(r"[^0-9.]", "", tok)
        if tok == "":
            return None
        # ドットが2つ以上（例: '2.0.0'）→ 最初の桁＋残りを小数部化
        if tok.count(".") >= 2:
            digits = tok.replace(".", "")
            if len(digits) == 1:
                return float(f"{digits}.0")
            return float(f"{digits[0]}.{digits[1:]}")
        # 先頭ドット（例: '.21'）→ 小数点左 1 桁落ちとみなし '2.1'
        if tok.startswith("."):
            d = tok[1:]
            if d == "":
                return None
            if len(d) == 1:
                return float(f"{d}.0")
            return float(f"{d[0]}.{d[1:]}")
        # ドット無し 2 桁以上（例: '17'）→ '1.7'
        if "." not in tok and len(tok) >= 2:
            return float(f"{tok[0]}.{tok[1:]}")
        # 1 桁のみ（例: '2'）→ '2.0'
        if "." not in tok:
            return float(f"{tok}.0")
        # それ以外は素直に float
        return float(tok)

    # 3) a:b 形式
    if ":" in s:
        a_raw, b_raw = s.split(":", 1)
        a = side_to_float(a_raw)
        b = side_to_float(b_raw)
        if a is None or b is None or a <= 0 or b <= 0:
            return ""

    # 4) コロン無しの代表パターン
    else:
        only = re.sub(r"[^0-9.]", "", s)
        if only == "":
            return ""
        # '1.21' / '1.2' / '1.2.0' 系
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
        # '0.xx'（割合表示）→ 比に復元し 1:X or X:1 にする
        elif re.fullmatch(r"0\.\d+", only):
            p = float(only)
            if p <= 0 or p >= 1:
                return ""
            X = (1.0 - p) / p  # = E/I
            # X≥1 で表記（1:X または X:1）
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

    # 5) ここから「1:X  or  X:1」・Xは1桁小数・X≥1 に強制
    #    （実値 a:b をスケールして一方を1に、もう一方を≥1に）
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

def Shift_Values(df, column):
    # 次の呼吸番号の最初の5フレームを抽出して最頻値を取得
    mode_values = {}
    grouped = df.groupby('BreathNumber')

    for breath, group in grouped:
        # 各呼吸の先頭5フレームを取得（5未満の場合はそのまま）
        first_values = group[column].head(5)
        # 最頻値（mode）を計算（複数ある場合は最初を使用）
        mode_val = first_values.mode()
        mode_values[breath] = mode_val.iloc[0] if not mode_val.empty else first_values.iloc[0]

    # 次呼吸の最頻値を取得してシフト
    next_values = pd.Series(mode_values).shift(-1)

    # 各行に対応するReal_値をマッピング
    return df['BreathNumber'].map(next_values)


def most_frequent(series):
    mode_values = series.dropna().mode()
    return mode_values.iloc[0] if not mode_values.empty else None


def _segment_by_cdyn(cdyn):
    """Cdynの値変化で呼吸境界(BreathNumber)を作る。
    連続するNaN(***＝Cdyn読取り不能)は「同一呼吸」とみなして束ねる。
    pandasでは NaN != NaN が True のため、素朴な ne(shift) だと***呼吸が
    1フレームずつ別セグメントに分裂し、後段の短呼吸マージで隣に吸収されて
    消滅する。連続NaNを境界にしないことで、***呼吸を独立した1呼吸として保持する。"""
    changed = cdyn.ne(cdyn.shift())
    changed &= ~(cdyn.isna() & cdyn.shift().isna())   # 連続NaN同士は変化なし扱い
    return changed.cumsum()


def CreateBreathNumber(df,
                       IECutoff=0.5,
                       ignore_first_m_within_breath=3,
                       ignore_last_k_within_breath=3,
                       N_for_moving_avg=30,
                       long_ratio=1.5,
                       short_ratio=0.5,
                       ie_split_min_delta=0.15):
    import pandas as pd
    import numpy as np

    df = df.copy()
    df["IE_Ratio"] = pd.to_numeric(df.get("IE_Ratio", np.nan), errors="coerce")
    if "Alert" not in df.columns:
        df["Alert"] = ""

    # ---- Step 1: Cdyn で大まかに分割 ----（連続NaN=***は1呼吸として束ねる）
    df["BreathNumber"] = _segment_by_cdyn(df["Cdyn"])

    # ---- Step 1.5: 短い呼吸(<=5フレーム)を隣接に統合（既存：IEベースの判定）----
    counts = df.groupby("BreathNumber").size()
    shorts = list(counts[counts <= 5].index)
    for b in shorts:
        g = df[df["BreathNumber"] == b]
        if g.empty:
            continue
        idxs = g.index.to_list()
        first, last = idxs[0], idxs[-1]

        prev_idx = df.index[df.index.get_loc(first)-1] if df.index.get_loc(first) > 0 else None
        next_idx = df.index[df.index.get_loc(last)+1] if df.index.get_loc(last) < len(df.index)-1 else None
        if prev_idx is None and next_idx is None:
            continue

        ie_cur = g["IE_Ratio"].mode().iloc[0] if not g["IE_Ratio"].dropna().empty else None
        ie_prev = df.at[prev_idx, "IE_Ratio"] if prev_idx is not None else None
        ie_next = df.at[next_idx, "IE_Ratio"] if next_idx is not None else None

        if ie_prev == ie_cur and ie_next != ie_cur:
            target_bn = df.at[prev_idx, "BreathNumber"]; target_cd = df.at[prev_idx, "Cdyn"]
        elif ie_next == ie_cur and ie_prev != ie_cur:
            target_bn = df.at[next_idx, "BreathNumber"]; target_cd = df.at[next_idx, "Cdyn"]
        else:
            if next_idx is not None:
                target_bn = df.at[next_idx, "BreathNumber"]; target_cd = df.at[next_idx, "Cdyn"]
            else:
                target_bn = df.at[prev_idx, "BreathNumber"]; target_cd = df.at[prev_idx, "Cdyn"]

        df.loc[idxs, "BreathNumber"] = target_bn
        df.loc[idxs, "Cdyn"] = target_cd

    # ---- 統合後に Cdyn が前後で一致していたらまとめる ----（同上：連続NaNは束ねる）
    df["BreathNumber"] = _segment_by_cdyn(df["Cdyn"])

    # ---- Step 2: IE_Ratio < IECutoff の変化で追加分割 ----
    # 分割はCdyn一定の区間内でのみ起き、IEの「変化」が信頼できる時だけ採用する。
    # （IE読取りの化け/空欄(0.0)との遷移や微小ジッタを境界と誤認すると、Cdyn一定の1呼吸を
    #   割ってFragmentを量産する。実データ照合でIE駆動Fragment 12件を解消・Cdyn由来16件は保持。）
    base_breath = df["BreathNumber"].copy()
    split_flag = pd.Series(0, index=df.index, dtype=int)

    for bnum, g in df.groupby(base_breath, sort=True):
        g_idx = g.index.to_list()
        if not g_idx:
            continue
        ier = g["IE_Ratio"]
        prev_ier = ier.shift(1)
        changed = ier.ne(ier.shift(1))
        changed.iloc[0] = False
        pos = pd.Series(range(len(g_idx)), index=g_idx)

        for i in g_idx:
            if not changed.loc[i]:
                continue
            n = len(g_idx); p = pos.loc[i]
            if p < ignore_first_m_within_breath: continue
            if p >= n - ignore_last_k_within_breath: continue
            v = ier.loc[i]; pv = prev_ier.loc[i]
            if not (pd.notna(v) and v < IECutoff):
                continue
            # 化け/空欄IE(0.0)との遷移では割らない（読取りの出現/消失は呼吸境界ではない）
            if not (pd.notna(pv) and pv > 0 and v > 0):
                continue
            # 微小なIEジッタでは割らない（信頼できる変化幅のみ境界とみなす）
            if abs(v - pv) < ie_split_min_delta:
                continue
            split_flag.loc[i] = 1

    df["BreathNumber"] = base_breath + split_flag.cumsum()

    # ---- Step 3: IECutoff 以上なら前に統合（連鎖対応）----
    merged = True
    while merged:
        merged = False
        bnums = sorted(df["BreathNumber"].unique())
        if len(bnums) <= 1: break
        for idx in range(1, len(bnums)):
            bn = bnums[idx]; prev_bn = bnums[idx-1]
            g = df[df["BreathNumber"] == bn]
            if g.empty: continue
            g = g.copy(); g["__pos__"] = range(len(g))
            g_valid = pd.to_numeric(
                g.loc[g["__pos__"] >= ignore_first_m_within_breath, "IE_Ratio"],
                errors="coerce"
            ).dropna()
            if not g_valid.empty and g_valid.iloc[0] >= IECutoff:
                df.loc[df["BreathNumber"] == bn, "BreathNumber"] = prev_bn
                merged = True; break
        if merged:
            mapping = {old: new for new, old in enumerate(sorted(df["BreathNumber"].unique()), start=1)}
            df["BreathNumber"] = df["BreathNumber"].map(mapping)

    # ========= NEW: Step 3.5 短呼吸の“後段マージ” (Cdyn基準) =========
    def _cdyn_rep(s: pd.Series) -> float:
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty: return np.nan
        m = s.mode(dropna=True)
        return float(m.iloc[0]) if not m.empty else float(s.median())

    counts = df.groupby("BreathNumber").size()
    short_bns = [int(b) for b, n in counts.items() if n <= 5]

    # 隣接探索に使う現在の並び
    def _neighbors(b):
        bnums = sorted(df["BreathNumber"].unique())
        if b not in bnums: return None, None
        i = bnums.index(b)
        prev_b = bnums[i-1] if i > 0 else None
        next_b = bnums[i+1] if i < len(bnums)-1 else None
        return prev_b, next_b

    for b in sorted(short_bns):
        g = df[df["BreathNumber"] == b]
        if g.empty: continue

        prev_b, next_b = _neighbors(b)
        if prev_b is None and next_b is None: continue

        cur_cdyn = _cdyn_rep(g["Cdyn"]) if "Cdyn" in g else np.nan
        prev_cdyn = _cdyn_rep(df.loc[df["BreathNumber"] == prev_b, "Cdyn"]) if prev_b is not None else np.nan
        next_cdyn = _cdyn_rep(df.loc[df["BreathNumber"] == next_b, "Cdyn"]) if next_b is not None else np.nan

        # 決定ロジック（Cdyn）
        atol = 1e-6
        to_prev = prev_b is not None and np.isfinite(prev_cdyn) and np.isfinite(cur_cdyn) and np.isclose(prev_cdyn, cur_cdyn, rtol=0, atol=atol)
        to_next = next_b is not None and np.isfinite(next_cdyn) and np.isfinite(cur_cdyn) and np.isclose(next_cdyn, cur_cdyn, rtol=0, atol=atol)

        if to_prev and not to_next:
            target = prev_b; rep_cdyn = prev_cdyn
        elif to_next and not to_prev:
            target = next_b; rep_cdyn = next_cdyn
        elif to_prev and to_next:
            # 両側同値 → 長い方へ
            prev_len = int(counts.get(prev_b, 0)); next_len = int(counts.get(next_b, 0))
            if prev_len >= next_len:
                target = prev_b; rep_cdyn = prev_cdyn
            else:
                target = next_b; rep_cdyn = next_cdyn
        else:
            # 近い側へ（|ΔCdyn|最小）。同値なら後側。
            candidates = []
            if prev_b is not None and np.isfinite(prev_cdyn): candidates.append(("prev", prev_b, abs(cur_cdyn - prev_cdyn), prev_cdyn))
            if next_b is not None and np.isfinite(next_cdyn): candidates.append(("next", next_b, abs(cur_cdyn - next_cdyn), next_cdyn))
            if candidates:
                candidates.sort(key=lambda x: (x[2], 0 if x[0]=="next" else 1))  # 同差なら next を優先
                _, target, _, rep_cdyn = candidates[0]
            else:
                target = next_b if next_b is not None else prev_b
                rep_cdyn = next_cdyn if next_b is not None else prev_cdyn

        # マージ適用
        idxs = g.index
        df.loc[idxs, "BreathNumber"] = target
        if "Cdyn" in df.columns and np.isfinite(rep_cdyn):
            df.loc[idxs, "Cdyn"] = rep_cdyn

        # counts は古いので更新（粗く再計算）
        counts = df.groupby("BreathNumber").size()

    # 後段マージ後の再連番
    mapping = {old: new for new, old in enumerate(sorted(df["BreathNumber"].unique()), start=1)}
    df["BreathNumber"] = df["BreathNumber"].map(mapping)
    if "__pos__" in df.columns:
        df.drop(columns="__pos__", inplace=True, errors="ignore")

    # ---- Step 4: Alert（Two_Breath / Fragment）----
    breath_lengths = df.groupby("BreathNumber").size().astype(int)
    if len(breath_lengths) > 0:
        window = max(1, min(N_for_moving_avg, len(breath_lengths)))
        moving_avg = breath_lengths.rolling(window=window, min_periods=1, center=True).mean()

        long_idx = breath_lengths.index[breath_lengths >= long_ratio * moving_avg]
        df.loc[df["BreathNumber"].isin(long_idx), "Alert"] = "Two_Breath"

        short_idx = breath_lengths.index[breath_lengths <= short_ratio * moving_avg]
        mask_short = df["BreathNumber"].isin(short_idx) & (df["Alert"] == "")
        df.loc[mask_short, "Alert"] = "Fragment"

    return df




# Cleaning OCRed text. Especially deleting noise like 2nd "."
def clean_decimal_number(text):
    """
    テキスト中の数字および小数点を抽出・整形し、floatまたはintとして返す。

    主な処理内容：
    - 数字および小数点以外の文字をすべて削除
    - 2つ以上ある小数点は最初の1つのみを残し、それ以降は削除
    - 最後が小数点で終わっている場合は削除
    - 整形後の文字列に応じて float または int 型に変換
    - 数字が存在しない・不正な場合は None を返す

    Parameters
    ----------
    text : str
        OCRなどで取得された文字列（例："6.2mg", "12..3", "abc7.89g"）

    Returns
    -------
    float or int or None
        正常に抽出・変換された数値（float または int）。変換できなければ None。

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



# ===== IE_Ratio 用ヘルパ（parse_IE_ratio の直上に追加） =====
import re, unicodedata

def _fix_round_X(x: float) -> float:
    """X を小数1桁に丸め。X>=10 は小数点誤読とみなし 1回だけ ÷10、X<1 は 1.0 へ。"""
    x = float(x)
    if x >= 10.0: x = x / 10.0      # 例: 21 → 2.1
    if x < 1.0:  x = 1.0
    return round(x + 1e-8, 1)

def _preclean_space_comma_to_colon(s: str) -> str:
    """空白/カンマ→コロン、全角→半角、連続コロン圧縮、端のコロン除去。"""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = (s.replace("・", ".").replace("_", ".").replace("：", ":")
           .replace(",", ":").replace("，", ":").replace("、", ":"))
    s = s.replace("|", " ")                      # 生文字連結の '|' を空白扱い（念のため）
    # 二重ドットはここで壊さない（先に余分な空白だけ除去）
    if re.search(r"\d+\.\d+\.\d+", s):
        s = re.sub(r"\s+", "", s)
        return s.upper()
    s = re.sub(r"\s+", ":", s)                   # 空白の連続 → コロン
    s = re.sub(r":{2,}", ":", s)                 # 連続コロン圧縮
    s = s.replace(":.:", ":")                    # '1 . 2.2' → '1:2.2'
    s = s.strip(":")                             # 先頭/末尾コロン除去
    s = re.sub(r"\s+", "", s)                    # 念のため
    return s.upper()

def _normalize_two_dot(s_pre: str) -> str:
    """a.b.c → 1:X / X:1 に正規化（X≥1, 小数1桁）。非該当は ''。"""
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
    """1:.xx / 1:.x.x… / .xx:1 / .x.x…:1 を 1:X / X:1 に復元。非該当は ''。"""
    if not s_pre or ":" not in s_pre:
        return ""
    # 1:.xx / 1:.x.x...
    m = re.fullmatch(r"\s*1\s*:\s*\.(\d+(?:\.\d+)*)\s*", s_pre)
    if m:
        d = m.group(1)
        digits = re.sub(r"\D", "", d)           # 例: "2.1" → "21"
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
    """既に 1:X / X:1 の形なら X を補正＆丸めして返す（X≥1, 1桁）。非該当は ''。"""
    if not s_pre or ":" not in s_pre:
        return ""
    # 余分なトークンが付いた場合の保険：最初の2トークンだけを見る
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
    """1.xx → 1:X（X≥1, 1桁）。非該当は ''。"""
    if not s_pre or ":" in s_pre:
        return ""
    m = re.fullmatch(r"1\.(\d+)$", s_pre)
    if not m:
        return ""
    d = m.group(1)
    X = float(f"{d[0]}.{d[1:]}") if len(d) >= 2 else float(f"{d}.0")
    return f"1:{_fix_round_X(X):.1f}"


# ===== ここから置換：parse_IE_ratio =====
def parse_IE_ratio(ie_raw):
    """
    IE_Ratio_raw → IE_Ratio（数値 I/(I+E)）に変換。
    仕様:
      - 純粋な小数（<1）は、その値をそのまま割合として採用（'.xx' も 0.xx と解釈）
      - それ以外は X:Y 形や >1 の小数（コロン誤認識）だけを対象に正規化
        前処理（空白/カンマ→コロン等）→ 二重ドット → 1:.xx/.xx:1 復元 → 既に 1:X/X:1 → 1.xx
        の優先順で 1:X / X:1（X≥1, 小数1桁、X≥10 は 1回だけ ÷10）へ整形し、 a/(a+b) を返す。
    """
    s_pre = _preclean_space_comma_to_colon(ie_raw)
    if not s_pre:
        return 0.0

    # 1) 純粋な小数（<1）はそのまま割合として採用（'.xx' も 0.xx）
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

    # 2) X:Y or >1 の小数 → 最小正規化で 1:X / X:1 に
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
    Edi領域だけ前処理：
      1) コントラスト強調（弱め）
      2) 平滑化（エッジを保ちつつノイズ低減）
      3) Otsu二値化（露出差への追従）
    入力: uint8 1ch（グレースケール）を想定
    """
    g = roi.copy()
    g = cv2.convertScaleAbs(g, alpha=2.0, beta=0)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    _, g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return g
