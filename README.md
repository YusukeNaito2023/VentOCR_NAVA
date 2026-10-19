# NAVA Engine (nava_ventocr)

## Version

1.5

### Changes in v1.5

- 解析の高速化（結果は v1.4 と同じ）: 読む範囲だけをグレースケール化・縮小する、EasyOCR の Reader を使い回す、
  後処理で使う画像を番号の小さい順に開く。1 時間の動画で v1.4 より 36% 速い。
- 詳細は CHANGELOG.md

### Changes in v1.4

- 解析の高速化: 後処理で全フレームの JPEG を読むのをやめた（engine_api 3）。異常フレーム（ポップアップなど）は initial_ocr の中で
  メモリにあるフレームで判定し、Frames.csv の `Abnormal` 列に記録する。
- 波形を読む開始・終了画像が Abnormal の呼吸だけ波形を書き出さない（v1.3 までは呼吸の途中に1枚でもあれば書き出さなかった）。
- 秒とフレーム数の換算に `GS.FPS` を使う（10 fps では v1.3 と同じ結果）。
- 詳細は CHANGELOG.md

### Changes in v1.3

- 波形の点の選び方を変更（`ExtractWaves._extract_waveform`）。v1.2 は列ごとに「色相が最も近い1画素」を
  採用しており、急峻な箇所（1列に縦線がある所）で選ぶ高さが偶然で決まり、画面に無いスパイクが出ていた。
  v1.3 は列ごとに「線が通った範囲」を取り、呼吸全体で最適な経路を引き、V字の谷・山の先端は残す。
- Flow の目盛りの推定で、Flow が急に変わる区間を当てはめから除き、`FLOW_GAIN` を 1.06 → 1.05 に。
- 平滑化オプション（smoothing）が効いていなかった不具合を修正（既定は OFF のまま）。
- 波形 CSV を呼吸の範囲（吸気の始まり〜次の吸気の始まり）に切り詰める（`trim_to_breath`、既定 ON）。
  前の呼吸の呼気の終わりと次の呼吸の立ち上がりを除き、時刻は Breaths.csv の Wave_T0 / Wave_Rate で保つ。
- 詳細は CHANGELOG.md

### Changes in v1.2

- 波形色のマッチングを BGR ユークリッド距離から **HSV(色相)ベース** に変更。
  色相は明度・彩度の低下に強いため、高解像度動画(1920x1080)を 1280x720 へ縮小した際に
  アンチエイリアスで暗く混色していた **Pressure の急峻な立ち上がり** を欠損なく検出できる。
  ROI・色定義・後処理インターフェースは v1.1 と互換。

## Overview

NAVA Engine is a waveform analysis module designed for Servo-i ventilator images.  
It extracts numerical parameters such as VTi, VTe, Cdyn, and Edi from waveform screenshots using OCR and image processing techniques, and structures them into breath-by-breath datasets.

This engine integrates:

- OCR-based value extraction (EasyOCR)
- Histogram-based trigger detection (OpenCV)
- Noise filtering and anomaly exclusion
- Breath segmentation and reconstruction

The engine is optimized for GPU acceleration.

---

## Architecture

### File Structure

```
Engine/NAVA/
├ core.py
├ ExtractWaves.py
├ sample_frame.jpg
└ README.md
```

---

## Dependencies

- EasyOCR (GPU supported)
- OpenCV
- pandas
- numpy
- tqdm
- re, gc

---

## Input

### Required Inputs

- `grayed_frame`  
  List of grayscale frames (OpenCV format)

- `total_frames`  
  Total number of frames

---

## Processing Pipeline

---

### 1. Initial OCR (`initial_ocr`)

#### 1.1 Preprocessing

- Load `sample_frame.jpg`
- Compute histogram for Triggered ROI
- Extract ROIs based on predefined coordinates (`Settings.ROIs`)

---

#### 1.2 ROI Processing

| ROI Name | Method |
|--------|------|
| Triggered | Histogram correlation (cv2.HISTCMP_CORREL) |
| VTi, VTe, Cdyn | EasyOCR → numeric cleaning |

---

#### Trigger Detection Logic

- Correlation > 0.9 → Not Triggered (0)
- Correlation ≤ 0.9 → Triggered (1)

---

#### OCR Normalization

- Remove multiple decimal points
- Empty → 0

---

#### Output

Frame-level DataFrame:

```
Frame | VTi | VTe | Cdyn | Triggered
```

---

### 2. Post Processing (`post_ocr`)

---

#### 2.1 Cdyn Correction

- If Cdyn > 100 → divide by 10
- Isolated values → set as NaN
- BreathNumber assigned at Cdyn transition points

---

#### 2.2 Mode Value Completion

For each breath:

- VTe → replaced with mode
- Edi Peak → replaced with mode

---

#### 2.3 Time Calculation

- Based on Frame index and FPS
- Format:

```
hh:mm:ss:cc
```

---

### 3. Breath Structuring

---

#### 3.1 Representative Frame Extraction

- First frame of each BreathNumber is used

---

#### 3.2 Filtering Short Breaths

- Frame difference ≤ 13 → excluded
- Merged into previous breath

---

#### 3.3 Real VTe Assignment

- Ventilator displays delayed VTe
- Assigned to previous breath as:

```
Real_VTe
```

---

### 4. Abnormal Detection

---

#### 4.1 AbnormalFrameDetector（initial_ocr の中）

- 各フレームの `ROI_LIST_Abnormal_Detection` の範囲を、区間の黒でないフレームの中央値と比べる（平均差・標準偏差の差・ヒストグラム相関）
- Frames.csv に記録:

```
Abnormal = 1
```

- 波形を読む開始・終了画像が Abnormal の呼吸は波形を書き出さない

---

#### 4.2 BreathsToCSVs()

- Each valid breath saved as:

```
BreathCSVs/1.csv
BreathCSVs/2.csv
...
```

---

## Output

---

### Frame-level Data

```
df_frames
```

---

### Breath-level Data

```
df_breath
```

Includes:

- Cdyn
- VTe
- Real_VTe
- BreathNumber

---

## Exception Handling

- OCR failure → 0 or NaN
- Trigger detection uses histogram (not OCR)
- Missing `sample_frame.jpg` → Exception

---

## Use Cases

- NAVA waveform analysis
- Edi signal extraction
- Preprocessing for annotation
- Research / development datasets

---

## Limitations

- OCR accuracy depends on image quality
- Requires correct ROI configuration
- Sensitive to UI layout changes

---

## Author

VentOCR Developer Team  
Contact: naito623@naramed-u.ac.jp
