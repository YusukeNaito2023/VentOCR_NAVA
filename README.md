# NAVA Engine (nava_ventocr)

## Version

0.2.0 (see CHANGELOG.md)

### Changes in 0.2.0

- Breaths are the ventilator's own boundaries (display updates of Cdyn / VTe) and are never merged or split by I:E. Up to 0.1.x, breaths whose Ti/Ttot was 0.5 or more (inverse-ratio I:E) were merged into the previous breath. Combining breaths is left to the combine_breaths module.
- Waveforms are built like the Hamilton engine: initial_ocr records the pen tip of every frame (Frames.csv `Sweep_X`, in pixels of the analyzed video), and each breath is rebuilt from its last image.
- Coordinates are written in pixels of the analyzed video (any frame size; the frames are read at 1280x720 internally). Up to 0.1.x the sweep positions were in 1280x720 coordinates, so the app drew the sweep line at 2/3 of the true position on 1920x1080 videos.
- Patient triggers come from the white part of the traces (the Servo-U draws the triggering part in white): Breaths.csv `Trigger` (A patient / C machine) and `Trigger_Source` (Pressure, Flow, Edi). The Triggered ROI is no longer OCRed (readtext could not read the trigger icon).
- Details in CHANGELOG.md

### Changes in 0.1.1 and 0.1.2

- 0.1.2: added the output declaration `outputs` to manifest.json (waves, breath_values, alerts; VentOCR 0.6.2 and later). The engine no longer uses Japanese (labels, messages, README, comments).
- 0.1.1: fixed analyses stopping with a KeyError when there are two or fewer breath boundaries (for example a video from another ventilator); such videos now finish with zero breaths.

### Changes in 0.1.0 (formerly v1.5.0)

- Faster analysis (same results as v1.4): grayscale and downscale only the regions that are read, reuse the EasyOCR Reader, and open the images used in post-processing in ascending frame order. 36% faster than v1.4 on a one-hour video.
- The engine key became `nava_ventocr` (earlier versions used `nava_ventocr_v1_X`) and versions are numbered 0.X.Y.
- Details in CHANGELOG.md

### Changes in v1.4

- Faster analysis: post-processing no longer reads the JPEG of every frame (engine_api 3). Abnormal frames (pop-ups, etc.) are detected inside initial_ocr from the frames in memory and recorded in the `Abnormal` column of Frames.csv.
- A breath's waveform is skipped only when the start or end image used to read it is Abnormal (up to v1.3, a single Abnormal frame anywhere in the breath skipped it).
- Seconds are converted to frame counts with `GS.FPS` (same results as v1.3 at 10 fps).
- Details in CHANGELOG.md

### Changes in v1.3

- Changed how waveform points are chosen (`ExtractWaves._extract_waveform`). v1.2 took the single pixel with the closest hue in each column, so at steep transitions (a vertical line in one column) the chosen height was essentially random and produced spikes that are not on the screen.
  v1.3 takes the vertical extent of the trace in each column, draws the optimal path over the whole breath, and keeps the tips of V-shaped valleys and peaks.
- The Flow scale estimate excludes segments where Flow changes steeply, and `FLOW_GAIN` changed from 1.06 to 1.05.
- Fixed the smoothing option having no effect (still OFF by default).
- Waveform CSVs are trimmed to the breath (inspiration onset to the next inspiration onset; `trim_to_breath`, ON by default).
  The end of the previous breath's expiration and the rise of the next breath are removed; timing is kept with Wave_T0 / Wave_Rate in Breaths.csv.
- Details in CHANGELOG.md

### Changes in v1.2

- Waveform color matching changed from BGR Euclidean distance to **HSV (hue) based** matching.
  Hue is robust to lower brightness and saturation, so the **steep rise of Pressure**, which became dark and blended by anti-aliasing when a high-resolution video (1920x1080) was downscaled to 1280x720, is detected without gaps.
  ROIs, color definitions and the post-processing interface are compatible with v1.1.

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

#### 4.1 AbnormalFrameDetector (inside initial_ocr)

- Compares the `ROI_LIST_Abnormal_Detection` regions of each frame with the median of the non-black frames of the segment (mean difference, standard-deviation difference, histogram correlation)
- Recorded in Frames.csv:

```
Abnormal = 1
```

- A breath whose start or end image (used to read the waveform) is Abnormal gets no waveform

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
