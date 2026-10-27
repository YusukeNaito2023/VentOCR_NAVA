# Base resolution (width, height)
# Input frames are normalized to this resolution before the ROIs are applied (resolution independence).
# The ROIs / ROI_LIST_Abnormal_Detection below are all defined in this resolution's coordinate system.
BASE_RESOLUTION = (1280, 720)

# Region of Interest
# Definition from NAVA_Layout_1.jpg
ROIs = {
    "VTe": [1139, 450, 55, 30],
    "EdiPeak":[953,518,67,31],
    "Cdyn": [1110,531,164,62],
    "IE_Ratio": [1086,243,168,44],
    # Displayed values used to convert waveforms to physical units (used by axis-scale estimation in ExtractWaves)
    "Ppeak": [952,112,80,70],     # The large "Ppeak" number (cmH2O)
    "VTi": [1139,382,55,30],      # The "VTi" number (ml). 68px above the VTe box
}

# 0.2.1: a 4-digit VTe / VTi (1000 ml or more) is wider than its ROI (about x 1136-1203), so its first and last digits
# were cut and EasyOCR read the cut digit as a letter ("1009" -> "10Oc" -> 10). When the template read of the ROI fails,
# these wider ROIs are read with their own templates (value_glyphs.npz sets "VTe_wide" / "VTi_wide"). The ROIs above
# are not widened: their templates read 3-digit values more often than templates made from the wide ROIs.
WIDE_ROIS = {
    "VTe": [1125, 450, 90, 30],
    "VTi": [1125, 382, 90, 30],
}


REFERENCE_IMAGE_PATH = 'normal.jpg' # This should be located in ./users/reference/

ROI_LIST_Abnormal_Detection = [
        (144, 91, 172, 631),     # ROI1: left-side region
        (1226, 92, 1251, 630)    # ROI2: right-side region
    ] # Where to search for comparing abnormal and normal frames


# 0.2.0: digit templates of the numeric ROIs (glyphs.py; built by tools/build_value_glyphs.py from the Servo-U 1-h video
# 2026-08-12 10-49-09, labels from EasyOCR, one set per ROI)
VALUE_GLYPH_FILE = "value_glyphs.npz"
