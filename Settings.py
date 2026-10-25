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
    "Triggered":[843,50,49,34],
    "IE_Ratio": [1086,243,168,44],
    # Displayed values used to convert waveforms to physical units (used by axis-scale estimation in ExtractWaves)
    "Ppeak": [952,112,80,70],     # The large "Ppeak" number (cmH2O)
    "VTi": [1139,382,55,30],      # The "VTi" number (ml). 68px above the VTe box
}


REFERENCE_IMAGE_PATH = 'normal.jpg' # This should be located in ./users/reference/

ROI_LIST_Abnormal_Detection = [
        (144, 91, 172, 631),     # ROI1: left-side region
        (1226, 92, 1251, 630)    # ROI2: right-side region
    ] # Where to search for comparing abnormal and normal frames
