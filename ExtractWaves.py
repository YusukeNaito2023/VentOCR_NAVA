import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import Global_Setting as GS
from . import Settings

# --- 欠損値補完とスムージングを行うかどうか ---

SMOOTHING = True
IMPUTATION = True

def set_config(settings):
    global SMOOTHING, IMPUTATION
    SMOOTHING = settings.get("post_ocr", {}).get("smoothing", True)
    IMPUTATION = settings.get("post_ocr", {}).get("imputation", True)

# --- 色と位置の定義 ---
COLOR_DEFINITIONS = {
    "Pressure": np.array([110, 163, 193]),
    "Flow": np.array([90, 123, 86]),
    "Volume": np.array([114, 118, 89]),
    "Edi": np.array([171, 105, 166])
}

Y_RANGES = {
    "Pressure": (88, 208),
    "Flow": (244, 339),
    "Volume": (380, 497),
    "Edi": (513, 633)
}

COLOR_TOLERANCE = 40





# --- 色距離に基づく波形抽出 ---
# imageを渡すと波形の時系列データを返してくれる内部関数
def _extract_waveform(img, target_bgr, max_dist=40, waveform_name=None, y_offset=0):
    h, w, _ = img.shape
    timeseries = []
    for x in range(w):
        column = img[:, x, :]
        global_x = x + globals().get("x_start", 0)
        target_bgrs = [COLOR_DEFINITIONS[waveform_name]]

        if 140 <= global_x <= 175:
            alt_colors = {
                "Pressure": [np.array([58, 115, 147])],
                "Flow": [np.array([53, 84, 47])],
                "Volume": [np.array([78, 82, 53])],
                "Edi": [np.array([82, 64, 93]), np.array([113, 75, 109]), np.array([129, 105, 135])]
            }
            target_bgrs = alt_colors.get(waveform_name, target_bgrs)
        elif 847 <= global_x <= 891 and waveform_name == "Edi":
            target_bgrs = [np.array([82, 64, 93]), np.array([113, 75, 109]), np.array([129, 105, 135])]

        dists_list = [np.linalg.norm(column - color, axis=1) for color in target_bgrs]
        dists = np.min(np.stack(dists_list), axis=0)

        mask = np.ones(len(dists), dtype=bool)
        if 140 <= global_x <= 175:
            background_color = np.array([17, 18, 14])
            text_gray_mask = (
                (column[:, 0] > 100) & (column[:, 0] < 230) &
                (np.abs(column[:, 0] - column[:, 1]) < 10) &
                (np.abs(column[:, 1] - column[:, 2]) < 10)
            )
            bg_dist = np.linalg.norm(column - background_color, axis=1)
            mask &= (bg_dist > 10) & (~text_gray_mask)
        else:
            background_color = np.array([37, 38, 34])
            bg_dist = np.linalg.norm(column - background_color, axis=1)
            mask &= bg_dist > 10

        dists[~mask] = np.inf

        # 判定: 通常色で検出成功
        if waveform_name == "Edi" and 847 <= global_x:
            timeseries.append(np.nan)
        elif np.min(dists) < max_dist:
            timeseries.append(h - np.argmin(dists))
        # 補助判定: 白色で再挑戦
        elif waveform_name in ["Pressure", "Flow", "Edi"]:
            white_color = np.array([219, 218, 209])
            white_dist = np.linalg.norm(column - white_color, axis=1)
            white_dist[~mask] = np.inf
            if np.min(white_dist) < max_dist:
                timeseries.append(h - np.argmin(white_dist))
            else:
                timeseries.append(np.nan)
        else:
            timeseries.append(np.nan)
    return timeseries

# --- 画像から時系列データを抽出 ---
def extract_waveform_timeseries(image_path, x_start=524, x_end=807):
    globals()["x_start"] = x_start
    img = cv2.imread(image_path)
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

    # Y範囲（3つのセグメント）
    y_ranges = [(88, 220), (224, 357), (363, 493)]

    # 画像読み込み
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

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
        waveforms = merge_waveforms_across_regions(img_end, regions)
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

def BreathsToCSVs(breaths, frames, frames_root, output_dir, user_id, overwrite=False, report_path=None):
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
    for _, row in tqdm(breaths.iterrows(), total=len(breaths)):
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

        # 波形抽出
        waveforms = merge_waveforms_across_regions(img_end, regions)
        length = len(next(iter(waveforms.values()), []))
        if length == 0:
            skip_logs.append({
                "BreathNumber": breath_num, "Status": "SKIPPED",
                "Reason": "empty waveform", "Details": "",
            })
            continue

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

        df_out = df_out.iloc[:-30]
        df_out.to_csv(out_path, index=False)
        print(f"{out_path} created")

        # サマリー/レポート
        summary_records.append({
            "BreathNumber": breath_num,
            "Start_Frame_Image": img_start,
            "End_Frame_Image": img_end,
            "Regions": "; ".join([f"{x0}–{x1}" for (x0, x1) in regions]),
        })
        skip_logs.append({
            "BreathNumber": breath_num, "Status": "OK", "Reason": "", "Details": ""
        })

    # ---- レポート出力 ----
    report_df = pd.DataFrame(skip_logs).sort_values("BreathNumber")
    report_df.to_csv(report_path, index=False)

    return len(summary_records), breaths


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


def DetectAbnormalFrames(user_id,df_frames):

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

    REFERENCE_IMAGE_PATH = os.path.join(GS.Dir_Frames(user_id),"00000", "frame_00000000.jpg")
    #REFERENCE_IMAGE_PATH = os.path.join(script_dir, "normal.jpg")


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
    ref_img = cv2.imread(REFERENCE_IMAGE_PATH)
    if ref_img is None:
        raise FileNotFoundError(f"正常画像が読み込めません: {REFERENCE_IMAGE_PATH}")

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
    df = df_frames
    results = []


    for _, row in tqdm(df.iterrows(), total=len(df)):
        folder_name = os.path.join(base_frame_dir, row["source_file"].replace("output_", "").replace(".csv", ""))
        file_name = "frame_" + str(int(row["iFrame"])).zfill(8) + ".jpg"
        filepath = os.path.join(folder_name, file_name)


        img = cv2.imread(filepath)

        if img is None:
            print(f"⚠️ 画像読み込み失敗: {filepath}")
            results.append(["読み込み失敗"])
            continue

        abnormal = False
        for i, (x1, y1, x2, y2) in enumerate(Settings.ROI_LIST_Abnormal_Detection):
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


        results.append(["1" if abnormal else "0",
                        mean_diff,
                        std_diff,
                        hist_corr])

    # 結果を保存
    df[["Abnormal", "MeanDiff", "StdDiff", "HistCorr"]]= pd.DataFrame(results)

    return df