# NAVA Engine (nava_ventocr)

## Version

1.1

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

#### 4.1 DetectAbnormalFrames()

- Detects noise / extreme values
- Marks as:

```
Abnormal = True
```

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
