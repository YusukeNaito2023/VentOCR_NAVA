# NAVA Engine (nava_ventocr)

## Version

0.2.2 (see CHANGELOG.md)

### Changes in 0.2.2

- Axis-label templates for the scales the Servo-U chooses by itself (it rescales Edi and Volume, and on some recordings the other bands, to fit the waveform): Edi 5-50, Volume 2-2250, Flow 8-150, Pressure 14-60, from 7 Servo-U recordings. A label may have several templates (recordings draw the labels slightly differently). Results are unchanged except axis scales that were estimated before.
- Edi under the Edi bar graph: the last sweep columns of the Edi band (x 847-888 at 1280x720) hold the Edi bar graph (purple, same hue as the trace) and a white dashed line, and were not read up to 0.2.1, so every breath drawn there had no Edi for about 0.5 s (329 of 1000 breaths of the Servo-U 1-h video). The trace is now read there as the pixels brighter than the bar (saturation >= 80, value >= 125); the dashed line's columns and columns without such pixels are filled linearly from both sides.
- Volume on the 0 line: at 0 ml the Servo-U draws the Volume trace along the bottom edge of the band, where it merges with the panel border and was not read, so Volume was empty after each expiration. A Volume gap whose read columns on both sides (also across the boundary to the next breath) are within 4 px of the band bottom is now 0; other gaps stay empty.

### Changes in 0.2.1

- Breaths.csv columns renamed: the breath's own value has the plain name (`VTe`, `VTi`, `Ppeak`, `Cdyn`, `EdiPeak`, `TiTtot`) and what the screen showed has the suffix `_displayed` (`VTe_displayed`, `Cdyn_displayed`, `EdiPeak_displayed`, `TiTtot_displayed`). Up to 0.2.0: `Real_VTe`, `Real_VTi`, `Real_Ppeak`, `Real_Cdyn`, `VTe`, `Cdyn`, `EdiPeak`, `IE_Ratio`. `VTi` / `Ppeak` are now the last reading of the breath, `TiTtot` is the value shown after the breath, and `EdiPeak` is empty when the screen held the previous breath's value or showed "***" (see 3.3). Frames.csv is unchanged.
- Fewer misread values: Cdyn is the value shown at the end of the breath (the display often changes 1-3 frames after VTe); EasyOCR reads that are not a plain number (including a space between digits) are not used; Ti/Ttot has extra digit templates and needs two decimals, except that a breath with no two-decimal read takes its "0.d" read as 0.d0.
- VTe / VTi of 1000 ml or more are read correctly: their 4 digits are wider than the ROI, so a wider ROI is read with its own templates when the ROI cannot be read.
- VTe / VTi reads above 5000 ml are not used (a frame caught while the display changes).
- VTe skips a value seen only on the breath's last frame (the display can show the next breath's VTe one frame early).
- New Alert Disconnect: VTe about 0 with VTi of 1000 ml or more and Cdyn "***" for 3 or more breaths (leak about 100%; possibly a disconnected circuit). Excluded from accuracy.
- The display-update frame and the 2 frames after it are kept for post_ocr too, so the app rarely has to make images.
- Waveform axis scales come from the axis labels on the breath's image where they can be read (`axis_labels.py`, templates `axis_labels.npz`); otherwise the estimate is used as before (see 4.2).

### Changes in 0.2.0

- Breaths are the ventilator's own boundaries (display updates of Cdyn / VTe) and are never merged or split by I:E. Breaths during which the display is not updated are not split. Up to 0.1.x, breaths whose Ti/Ttot was 0.5 or more (inverse-ratio I:E) were merged into the previous breath. Combining breaths is left to the combine_breaths module.
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
- Patient-trigger detection from the white part of the traces (OpenCV)
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
├ glyphs.py, value_glyphs.npz        (digit templates of the numeric ROIs)
├ axis_labels.py, axis_labels.npz    (templates of the waveform axis labels, 0.2.1; more labels in 0.2.2)
├ tools/build_value_glyphs.py, tools/build_axis_labels.py
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
| VTi, VTe, Cdyn | EasyOCR → numeric cleaning |

---

#### Trigger Detection Logic (post_ocr, BreathsToCSVs)

- The Servo-U draws the triggering part of a trace in white (Paw for a pressure trigger, Flow for a flow trigger, Edi for an Edi trigger in NAVA mode).
- For each breath, white pixels (saturation ≤ 45, brightness ≥ 100) within 4 rows of the extracted Paw / Flow / Edi trace are counted per column, from 0.4 s before the inspiration onset (`TRIG_LEAD_COLS` = 33 columns; a pressure trigger during expiration comes before the flow onset) to 1.0 s after it. The top 22 rows of each band (panel titles) and the pen tip are not used.
- White screen furniture is not counted: pixels white on at least 30% of every 5th image read (grown by 2 px), such as the scale numbers at the bottom-left of each band and the Edi bar graph. This works for any layout and frame size.
- A band with at least 2 such columns is a trigger source: `Trigger` = A (patient) / C (machine), `Trigger_Source` (e.g. Pressure, Flow, Edi, Pressure+Edi), counts `Trig_White_P/F/E`.
- Checked against the Servo Tracker export (Trig cause): Servo-U 1 h in PC mode 703/705 patient, 282/282 machine; flow-trigger clip 99/103, 90/90; NAVA-mode clip 46/48 pressure and 9/13 Edi triggers, 115/115 machine.

---

#### Template Reading

- The numeric ROIs are read on every frame by digit template matching (`glyphs.py`, templates `value_glyphs.npz`, one set per ROI). EasyOCR is used only when the templates cannot read an ROI, and its read is used only when it is a plain number, with no space between digits (0.2.1). A digit may have a second template for a slightly different drawing (0.2.1: Ti/Ttot; `tools/build_value_glyphs.py --add-variants`). VTe and VTi also have a wider ROI with its own templates (`Settings.WIDE_ROIS`, sets `VTe_wide` / `VTi_wide`), read when the ROI cannot be read, because 4-digit values are wider than the ROI (0.2.1; `tools/build_value_glyphs.py --from-rois`). New templates (another ventilator or screen) can be built with `tools/build_value_glyphs.py` from an analysis of a video of that screen.

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

#### 3.1 Breath Boundaries

- A breath starts at a display update of Cdyn or VTe (`BOUNDARY_KEYS`). A new value must stay for at least 0.3 s (`STABLE_SEC`); shorter changes are misreads. Updates of Cdyn and VTe within 0.5 s (`MERGE_SEC`) are one boundary.
- Breaths are boundaries only where the displayed values change. Breaths during which the display is not updated (for example while VTe is about 0 and Cdyn shows "***", when every breath shows the same values) are not split: they stay inside one breath. No breaths are added from the waveforms.
- Breaths are never merged or split by I:E (combining breaths is left to the combine_breaths module).
- Frames.csv `BreathMark` = 1 at the first frame of each breath; `BreathNumber` counts them.

---

#### 3.2 Alerts

Breaths.csv / Frames.csv `Alert` (comma separated):

| Alert | Meaning |
|---|---|
| Fragment | First or last breath, cut by the start or end of the recording |
| Two_Breath | At least 1.5x the neighbouring breaths: possibly breaths without a display update in between (not split, see 3.1) |
| Short | At most half the neighbouring breaths: possibly a misread display update |
| Long | Longer than one sweep: only the part still on the screen is written (`Wave_T0` is the oldest column on the screen, not the inspiration onset) |
| Partial | The end of the waveform is not on the screen yet |
| Abnormal | Pop-up or another screen on the image of the breath (no waveform) |
| Disconnect | 0.2.1: the breath's VTe ≤ 5 ml, VTi ≥ 1000 ml and Cdyn "***" on more than half of its frames, for 3 or more breaths in a row (one non-matching breath inside a run is included). Leak about 100%, possibly a disconnected circuit. Excluded from accuracy |

---

#### 3.3 Breath values (Breaths.csv, 0.2.1)

- The ventilator shows some values of a breath only after it: VTe and Cdyn at the start of the next breath, Ti/Ttot shortly after the breath ends. VTi and Ppeak are on the screen at the end of the breath (Ppeak when the pressure peak is reached, VTi when the breath turns to expiration).
- The breath's own value has the plain name; what the screen showed during the breath has the suffix `_displayed`:

| Column | Value |
|---|---|
| `VTe`, `Cdyn` | Shown at the start of the next breath (mode of its first 5 frames; Cdyn "***" -> empty) |
| `VTi`, `Ppeak` | Last reading of the breath |
| `TiTtot` | `TiTtot_displayed` of the next breath |
| `EdiPeak` | `EdiPeak_displayed`, but empty when it equals the previous breath's (the screen held the value) or the screen showed "***" |
| `VTe_displayed` | Last reading of the breath, except a value seen only on its last frame (often the previous breath's VTe) |
| `Cdyn_displayed`, `EdiPeak_displayed` | Shown at the end of the breath ("***" -> empty) |
| `TiTtot_displayed` | Last reading of the breath (Ti/Ttot = I/(I+E)) |

- Where an axis label cannot be read (see 4.2), the waveform axis scale is estimated from the VTi / Ppeak shown at the start of the next breath (as up to 0.2.0; not written) and `EdiPeak` (breaths with an empty `EdiPeak` take the previous breath's Edi scale).
- Up to 0.2.0 the names were `Real_VTe`, `Real_Cdyn`, `Real_VTi`, `Real_Ppeak` (VTi / Ppeak from the first 5 frames of the next breath), `VTe`, `Cdyn`, `EdiPeak` and `IE_Ratio`, and the breath's first frame's VTi / Ppeak and OCR text / confidence (`*_raw`, `*_conf`, still in Frames.csv) were also written.

---

### 4. Abnormal Detection

---

#### 4.1 AbnormalFrameDetector (inside initial_ocr)

- Compares the `ROI_LIST_Abnormal_Detection` regions of each frame with the median of the non-black frames of the segment (mean difference, standard-deviation difference, histogram correlation)
- Recorded in Frames.csv:

```
Abnormal = 1
```

- A breath whose image (its last frame, or a later frame still showing it) is Abnormal gets no waveform (Alert Abnormal)

---

#### 4.2 BreathsToCSVs()

- Axis scales (0.2.1, `P_min` / `P_max`, `F_min` / `F_max`, `V_max`, `E_max`): the labels at the left edge of each band
  (top of every band, bottom of Pressure and Flow) are read on the image the breath's waveform is read from
  (`axis_labels.py`). At 1280x720 the digits are 7-8 px tall and blurred, so each label is compared as a whole with
  the labels in `axis_labels.npz` (RMS difference of the gray levels; the same label about 1.5, a different label 7.6
  or more, also one without a template). A label is used only when its best template is within 4.5 and the next best
  is at least 3 further away; pixels of a waveform crossing the label are left out (at most 25% of the box). A wave
  whose top label is not read keeps the estimate (from Ppeak / VTi / EdiPeak and the waveforms, as up to 0.2.0),
  except Flow, whose scale is symmetric on the Servo-U: a read bottom label gives the top (max = -bottom). A
  bottom label that is not read comes from the 0 baseline. Labels without a template (other scales, other screens)
  are never read; add them with `tools/build_axis_labels.py` (extract key frames of a video, group the label boxes,
  name the groups by eye, build). A label may have several templates (0.2.2; recordings draw the labels slightly
  differently); a read then needs the gap of 3 to the best template of a different label. About 4 ms per breath.
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

Includes (see 3.3):

- BreathNumber
- VTe, VTi, Ppeak, Cdyn, EdiPeak, TiTtot
- VTe_displayed, Cdyn_displayed, EdiPeak_displayed, TiTtot_displayed
- Trigger, Trigger_Source, Wave_T0, Wave_Rate, Alert

---

## Exception Handling

- OCR failure → 0 or NaN
- Trigger detection uses the white part of the traces (not OCR)
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
- Axis labels are read only for the labels in `axis_labels.npz` (made from 7 Servo-U recordings); other scales and recordings that draw the labels differently use the estimate

---

## Author

VentOCR Developer Team  
Contact: naito623@naramed-u.ac.jp
