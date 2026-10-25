# CHANGELOG

## 0.2.0 (unreleased)
Results change (breath boundaries, waveforms, new columns). Same approach as the Hamilton engine.

### Breath boundaries
- A breath starts where the ventilator updates its per-breath displays: Cdyn or VTe shows a new value (or "***") for at least 0.3 s; updates within 0.5 s count as one boundary. The two displays change together at the start of each breath; VTe also splits breaths where Cdyn stays the same or shows "***".
- Breaths are no longer merged or split (CreateBreathNumber Steps 1.5-3.5 were removed). Up to 0.1.x, Step 3 merged every breath whose Ti/Ttot (I:E as I/(I+E)) was 0.5 or more into the previous breath; on a one-hour Servo-U recording this merged 87 breaths that the ventilator had counted (inverse-ratio I:E such as "6.3 : 1"). Combining breaths is left to the combine_breaths module.
- Boundaries come only from changes of the displayed values. Breaths during which the display is not updated (for example while VTe is about 0 and Cdyn shows "***", when every breath shows the same values) are not split, and no breaths are added from the waveforms. On the one-hour Servo-U recording this leaves 41 of the ventilator's breaths inside longer breaths.
- Alerts: Fragment (first / last breath, cut by the recording), Two_Breath (at least 1.5x the moving average of breath lengths), Short (at most half). They only flag breaths.

### Waveforms (rebuilt from the last image)
- initial_ocr records the pen tip of every frame (`sweep.SweepDetector`: a few rows of the Pressure / Flow / Volume bands compared with the previous frame; the newly drawn trace and the erase band about 38 px ahead give the tip). post_ocr tracks it at constant speed (`sweep.fit_sweep`) into an unwrapped position.
- Each breath runs from its inspiration onset (found in Flow from 1.0 s before to 0.5 s after the display update) to the next breath's onset, and is read from one image: the last frame of the breath (a later frame if the next onset is still ahead of the pen). The Servo-U redraws the whole area at each wrap (scales may change), so one image has one scale. Breaths longer than one sweep keep only the part still on the screen (Alert Long).
- Images are opened in ascending order first (no seeking in the app's segment video).
- The white fallback of the trace detection skips the panel title rows and keeps every white run as a candidate. It used to take the topmost white pixel, which picked the title text above a white trigger mark and drew a spike.
- Up to 0.1.x each breath used a start image and an end image (and the frame before each) and their difference.

### Coordinates
- Written in pixels of the analyzed video (frames of any size are read at 1280x720 internally): Frames.csv `Sweep_X` (pen tip on every frame) and the waveform rows (1 row = 1 column of the video; values read at 1280x720 are interpolated onto those columns), with `Wave_T0` / `Wave_Rate`. Breaths.csv no longer has X_Start / X_End / Sweep_Cursor_Start / Sweep_Cursor_End, which were in 1280x720 coordinates and made the app draw the sweep line at 2/3 of the true position on 1920x1080 videos.

### Patient trigger
- The Servo-U draws the triggering part of a trace in white. White pixels (saturation ≤ 45, brightness ≥ 100: thin marks are dimmer after downscaling) on the extracted trace (within 4 rows, excluding the panel title rows; excluding white screen furniture: pixels white on at least 30% of every 5th image read, grown by 2 px, such as the scale numbers at the bottom-left of each band and the Edi bar graph) from 0.4 s (33 columns) before to 1.0 s after the onset (a pressure trigger during expiration comes before the flow onset), in the Pressure / Flow / Edi bands: Breaths.csv `Trigger` (A patient / C machine), `Trigger_Source` and the counts `Trig_White_P/F/E`.
- Checked against the Servo Tracker export (Trig cause): Servo-U 1 h in PC mode, patient 703/705 and machine 282/282; a flow-trigger clip, 99/103 and 90/90; a NAVA-mode clip, pressure 46/48, Edi 9/13 and machine 115/115.
- The Triggered ROI (patient-trigger icon) is no longer OCRed: up to 0.1.x it was read with readtext every 10 frames, which never returned text (all NaN) and was not used.

### Other
- IE_Ratio (Ti/Ttot) of a breath is the last value read in the breath (it was the first frame's). The display sometimes updates 1-2 frames after Cdyn / VTe, so the first frame often showed the previous breath's value. On the one-hour Servo-U recording, agreement with the ventilator export rose from 73.6% to 81.6% (direct-decode frames: 69.0% to 72.5%).
- Diff gating: an ROI is OCRed again when at least 0.5% of its pixels changed by 24 gray levels or more since its last OCR (engine options `diff_px_level` 24, `diff_px_frac` 0.005), the same for every ROI. Up to 0.1.x the mean absolute difference was used (`diff_thresh` 2.0, `diff_thresh_cdyn` 0.5, removed): a one-digit change of Ti/Ttot moves it by only about 1.3, so on a Servo-U test of 4 minutes 393 frames kept a stale Ti/Ttot; and on frames decoded straight from the video (no re-encoding), noise of 0.6-1.3 made Cdyn be OCRed on about 92% of frames. With the new rule: OCR on 1-4% of frames per ROI and no stale frames.
- EdiPeak preprocessing runs only for ROIs that are OCRed (the diff gating compares the ROI before preprocessing).
- Settings removed: similarity_thresh, sampling_size, detect_sweep_line.
- Alert Abnormal: the image of the breath is a pop-up or another screen (no waveform).
- manifest: Trigger / Trigger_Source in breath_values; the I:E value is labelled Ti/Ttot (it is I/(I+E), e.g. "1 : 2.3" -> 0.30).

### Checks (so far)
- Servo-U recording, first 3 minutes (1920x1080), against the ventilator's own export (Servo Tracker, Breath_*.csv; video time + 330.25 s = log time): 64 of 64 breaths have an exported breath within 0.5 s (median 0.09 s); 1 exported breath missing (the first, at 4.4 s). Trigger agrees for 64 of 64 (export Trig cause 4 = Pressure trigger, 2 = machine).

## 0.1.2 (2026-10-25)
Analysis results are the same as 0.1.1. Added the output declaration `outputs` to manifest.json (VentOCR 0.6.2 and later; ENGINE_API.md §8.1; same structure as the Hamilton engine). `ventocr_min` is now 0.6.2.
- waves: Pressure (label Paw, cmH2O), Flow (L/min), Volume (mL), Edi (µV). The y-axis uses the on-screen scale (P/F/V/E_min/max). Colors follow the app's common palette
- breath_values (order = order in the breath properties):
  - Real_VTe (label VTe), Real_VTi (VTi), Real_Ppeak (Ppeak): values of this breath (the display at the start of the next breath)
  - EdiPeak: value of this breath (the display updates in the middle of the breath)
  - IE_Ratio (I:E ratio, the value of I/E)
  - VTe (displayed), Cdyn (displayed): values shown during this breath = values of the previous breath (the Cdyn and VTe displays update at the breath boundary)
  - Display update timing check (sample, 45 breaths): Cdyn and VTe changed at the start of the breath all 46 times; VTi, Ppeak and EdiPeak change mid-breath (30-40% into the breath)
- alerts: Two_Breath (at least 1.5x the neighbouring breaths) and Fragment (at most half of them). Both are excluded from the accuracy evaluation
- The engine no longer uses Japanese: manifest labels, progress/error/log messages, README, code comments and docstrings, .gitignore and calibration_template.csv are in English. Full-width characters used to normalize OCR text are kept. Entries for earlier versions below remain in Japanese

## 0.1.1 (2026-10-21)
### 不具合の修正: 呼吸が少ないと解析が止まる
- 波形を書き出す候補の呼吸が無いとき（呼吸の区切りが 2 つ以下。機種の確認で「続行」を選んだ合わない動画など）、
  `_BreathsToCSVs` の最後で、空の `skip_logs` から作った列の無い表を `sort_values("BreathNumber")` して KeyError になり、解析が失敗していた
- 空でも列（BreathNumber・Status・Reason・Details）を持つ表にする。末尾の呼吸を除く処理も、呼吸が無いときに落ちない形（`iloc[:-1]`）に
- 呼吸が 3 つ以上ある動画の結果は 0.1.0 と同じ
- 確認: Cdyn が読めない表・区切り 1〜4 個の表を post_ocr に渡して、すべて最後まで通る（0.1.0 は区切り 2 個以下で KeyError）。
  Hamilton の動画（20 秒）を NAVA で解析して、0 呼吸のまま正しく終わる（Breaths.csv 0 行・波形なし）

## 0.1.0 (2026-10-19)
旧 v1.5.0 と中身は同じ。エンジンキーを nava_ventocr にまとめ（v1.1〜v1.5 は版ごとに nava_ventocr_v1_X だった）、版の付け方を 0.X.Y に改めた（古い版のタグは削除）。

## 1.5.0 (2026-10-19)
v1.4 を複製して作成（エンジンキー `nava_ventocr_v1_5`）。解析の高速化だけで、**結果は v1.4 と同じ**（Frames.csv・Breaths.csv・波形 CSV を比べて一致）。

### 区間の処理（initial_ocr）
- 読む範囲だけをグレースケール化する（`_GrayView`）。v1.4 までは全画面を float64 で計算していた（1 フレーム約 8 ms）。式は同じなので結果も同じ
- 1920x1080 の動画は、全画面を 1280x720 へ縮小せず、読む範囲だけを縮小する（`_BaseFrame`）。縮小率 3:2 の周期にそろえた範囲を
  INTER_AREA で縮小すると、全画面を縮小したのと同じ画素になる。周期にそろえられない解像度は v1.4 と同じく全画面を縮小する
- EasyOCR の Reader を同じプロセスの中で使い回す（`_READER_CACHE`）。v1.4 までは区間ごとにモデルを読み直していた（1 回約 1.8 秒）
- 区間（60 秒・600 フレーム）の initial_ocr: 1080p 12.5 → 3.4 秒、720p 7.9 → 2.5 秒

### 後処理（BreathsToCSVs）
- 使う画像（開始・終了とそれぞれの直前）を、呼吸ごとの処理の前に区間・フレーム番号の小さい順に開いておく。
  アプリ（engine_api 3）は開かれた画像を区間動画から作り、前へ進む読み方なら位置を指定し直さずに読み進められる。
  v1.4 は呼吸ごとに終了 → 開始の順（後ろへ戻る）に開いていたので、そのたびに区間動画のシークが起きていた

### 確認
- 1 時間の動画（Servo-U、1920x1080、36,000 フレーム、985 呼吸。VentOCR b97dc00、並列 5、M3）:
  v1.4 955.7 秒（区間の処理 608.9・後処理 345.3）→ v1.5 611.2 秒（478.2・131.7）、−36%。
  Frames.csv・Breaths.csv（File_Name 以外）・波形 CSV 981 本・BreathCSV_Report.csv が一致
- 読む範囲だけの縮小・グレースケール化は、1080p・720p・4K・周期にそろわない解像度の 378 か所で、全画面で計算したものと画素が一致

## 1.4.0 (2026-10-18)
v1.3 を複製して作成（エンジンキー `nava_ventocr_v1_4`）。解析の高速化のため、後処理で全フレームの JPEG を読むのをやめた。
`manifest.json` の `engine_api` を 3 に（VentOCR 0.5.0 から。アプリは全フレームの画像を保存せず、エンジンが開いたフレームだけを作る。
ENGINE_API.md §5.1）。波形の抽出のしかたは v1.3 と同じ。

### 異常フレーム（ポップアップなど）の判定を initial_ocr へ
- v1.3 までは post_ocr の `DetectAbnormalFrames` が全フレームの JPEG を読み直していた（サンプル 1205 フレームで全部。1 時間の動画で 36,000 枚）
- v1.4 は initial_ocr の中で、メモリにあるフレームで判定する（`ExtractWaves.AbnormalFrameDetector`）。Frames.csv の列
  `Abnormal` / `MeanDiff` / `StdDiff` / `HistCorr` は v1.3 と同じ
- 判定の範囲（`ROI_LIST_Abnormal_Detection`）と閾値は v1.3 と同じ。基準は「動画の先頭の黒でないフレーム1枚」から
  「区間（initial_ocr の単位）の黒でないフレームの画素ごとの中央値」に（区間ごとに動くため。区間の先頭がポップアップでも崩れない）。
  区間がすべて黒画面なら、その区間は Abnormal
- `DetectAbnormalFrames` と `post_ocr.abnormal_workers` は削除（古い設定に残っていても無視する）

### 波形を書き出さない呼吸
- v1.3: 呼吸の中に1枚でも Abnormal があれば書き出さなかった
- v1.4: 波形を読む開始画像・終了画像が Abnormal のときだけ書き出さない（BreathCSV_Report.csv の Details は `start=0, end=1` など）。
  呼吸の途中のポップアップは問わない（終了画像から波形が読めればよい。Hamilton と同じ考え方）

### フレームの画像の読み方（engine_api 3）
- 読むのは呼吸ごとに開始・終了画像とそれぞれの直前の画像（掃引の位置の検出）の4枚だけ。`cv2.imread` にパスを渡して読む
- 画像があるかの確認は `os.path.exists` → `os.path.isfile`
- BreathsToCSVs の間は直近 8 枚の画像を覚えておき、同じ画像を読み直さない（終了画像は3回読んでいた）
- 診断用の `detect_sweep_line`（既定 OFF）は、ON にすると今までどおり全フレームを読んで書き換える

### フレームレート（engine_api 2）
- 決め打ちのフレーム数を秒から `GS.FPS` で求める（10 fps では v1.3 と同じ）: 開始画像は区切りの 0.6 秒前（6 フレーム）、
  短い呼吸の統合は 0.5 秒以下（5 フレーム）、IE 比の判定で呼吸の先頭・末尾の 0.3 秒（3 フレーム）を除く

### 確認
- サンプル（nava_ventocr_v1_2_sample.mp4、10 fps・1205 フレーム）: Breaths.csv・Frames.csv（判定の数値の列以外）・波形 CSV 45 本が
  v1.3 と一致。開いたフレームの画像は 1205 枚 → 180 枚（延べ 180 回）
- 同じ動画を 20 fps: 45 呼吸、値・呼吸の長さは 10 fps と同じ
- ポップアップを合成した動画（呼吸 10 の途中に 1 秒、呼吸 20 の終了画像に 0.6 秒）: Abnormal のフレームは v1.3 と同じ 16 枚。
  v1.3 は呼吸 10・20・21 を書き出さなかったが、v1.4 は呼吸 20 だけ。呼吸 10・21 の波形はポップアップの無い解析と一致

## 1.3.0 (2026-10-11)
v1.2 を複製して作成（端末版）。変更は `ExtractWaves.py` と、core.py の `ENGINE_VERSION`（v1.2 でも "1.1.0" のままだった）→ "1.3.0"。

### 波形の点の選び方（`_extract_waveform`）
- 不具合: 画面に無いスパイクが Pressure / Flow / Volume に出る（例: サンプル B10 の t≈0.73s、吸気→呼気の切り替わり）
- 原因: v1.2 は列ごとに「色相が最も近い1画素」を採用していた。急峻な箇所では1列に 30〜60px の縦線があり
  （縮小・JPEG のにじみで数列に広がる）、色相の差は 0〜2 しかないので、選ばれる高さがほぼ偶然で決まり、
  隣の列と上下に跳ねていた
- 対策: 列ごとに「線が通った範囲」を取る
  1. 縦に続いた画素のかたまり（`RUN_JOIN` 未満の切れ目はつなぐ）から、隣の列とつながる1本の線を
     呼吸全体で最適に選ぶ（`_choose_runs`、Viterbi）。孤立したノイズや文字の画素を除ける
  2. 線の太さを補正した範囲の中で、なめらかな経路を呼吸全体で最適に引く（`_smooth_path`）
  3. 範囲が経路から局所的に突き出している所（V字の谷・山の先端）はその値に固定して引き直す（`_find_tips`）
- 画素の判定（色相・彩度・明度・左端の文字除外・白いペン先）は v1.2 と同じ
- 確認（サンプル 45 / 0722-1313 79 / 2026-06-01 の最初の10分 137 呼吸）:
  - 偽スパイクのある呼吸（P/F/V）: サンプル 14/34/41 → 0/2/0。目視で B10 などのスパイクが消えたことを確認
  - Flow の呼気の V 字の底、圧の行き過ぎ（落ち込み直後の谷・揺れの1つ目の山）を残す（v1.2 は取り逃していた）
  - v1.2 が1列だけ拾っていた外れ画素（底が 1〜9px に落ちる）を拾わない
  - ピーク画素は 0.5〜1.5px 低い（線の上端ではなく中心を取るため）
- 残る弱点: 揺れが何度も続く場合、2つ目以降の小さな谷は少しつぶれる。V字の底が左端の目盛りの文字に
  重なると、1列だけ文字を拾うことがある

### 縦軸の目盛り（Flow）
- Flow の積分と Volume の増減の当てはめから、Flow が隣の列と `FLOW_STEEP_PX`（3px）を超えて変わる区間を除く。
  先端を残すようになって当てはめがばらついたため
- `FLOW_GAIN` 1.06 → 1.05
- 画面の目盛りとの一致（3本 261 呼吸）: v1.2 206 → v1.3 213。Volume は 2026-06-01 で 104 → 109 / 137。Pressure は変化なし

### 波形を呼吸の範囲に切り詰める（`_trim_records_to_breaths`、`post_ocr.trim_to_breath`、既定 ON）
- 以前からの現象: 波形 CSV の先頭に前の呼吸の呼気の終わり（約 0.4〜0.6 秒）、末尾に次の呼吸の吸気の
  立ち上がりが入る。呼吸の区切り（Cdyn の表示の更新）が吸気の始まりと微妙にずれても呼吸を途中から
  描かないよう、開始画像を区切りの 0.6 秒前から取っているため
- Flow の吸気の立ち上がり（基線から 3px 超が 5 列以上続き、8px 以上に届き、その前に吸気でない列が 5 列以上）を
  見つけ、立ち上がりの足元（最大値の 20% を超えた点から下がり続ける所までさかのぼる）の 3 列前から残す
  - 先頭: 最初の吸気の始まりまで削る。先頭・末尾の空の行も削る
  - 末尾: 呼気を伴わない最後の吸気（次の呼吸の吸気が途中で切れたもの）の始まりまで削る。
    2呼吸ぶん入った呼吸の2つ目は残す（消すとどの CSV にも残らないため）
  - 次の呼吸で見つけた吸気の始まりの時刻で、前の呼吸の末尾も切る（短すぎて自分では判定できない立ち上がりを除く）。
    自分の吸気の始まりが無い呼吸（壊れた呼吸）は切らない
  - 欠けた列（掃引の折り返しの継ぎ目）は判定のときだけつなぐ。欠ける直前が呼気なら偽の立ち上がりとして使わない
- 削った分の時刻は Breaths.csv の `Wave_T0`（最初の点の時刻・秒）/ `Wave_Rate`（px/秒）で保つ（アプリの
  sample_times がこれを優先する）。`Wave_T0` はアプリと同じ求め方（開始画像のペン先から逆算）。
  削った行数は `Wave_Trim_Head` / `Wave_Trim_Tail`。CSV の X は元の列番号のまま
- 目盛りの推定は切り詰める前の波形で行う。計算は1呼吸あたり約 0.07 ms
- 確認: サンプルで隣の呼吸との重なりが中央値 0.73 秒 → 0（最大 0.00 秒）、同じ点の時刻は切り詰め前と一致。
  0722-1313 は掃引位置とフレームの時刻が少し食い違っており、以前は次の呼吸の先頭が重なって隠れていた
  約 0.3 秒の隙間が見えるようになった（データは欠けていない）
- 再抽出（`re_extract_breath_waveform`）は時刻の情報が無いので切り詰めない

### その他
- 平滑化オプション（`post_ocr.smoothing`）が効いていなかった不具合を修正（移動平均の結果を直後に上書きしていた）。
  既定は OFF（ON にすると先端が削れる）
