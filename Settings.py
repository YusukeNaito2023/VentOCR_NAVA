# 基準解像度 (width, height)
# 入力フレームはこの解像度に正規化してから ROI を適用する（解像度非依存化）。
# 下記の ROIs / ROI_LIST_Abnormal_Detection はすべてこの解像度の座標系で定義する。
BASE_RESOLUTION = (1280, 720)

# Region of Interest
# Definition from NAVA_Layout_1.jpg
ROIs = {
    "VTe": [1139, 450, 55, 30],
    "EdiPeak":[953,518,67,31],
    "Cdyn": [1110,531,164,62],
    "Triggered":[843,50,49,34],
    "IE_Ratio": [1086,243,168,44],
    # 波形を物理単位へ換算するための表示値（ExtractWaves の軸スケール推定で使う）
    "Ppeak": [952,112,80,70],     # 「Ppeak」の大きな数値（cmH2O）
    "VTi": [1139,382,55,30],      # 「VTi」の数値（ml）。VTe 枠の 68px 上
}


REFERENCE_IMAGE_PATH = 'normal.jpg' # This should be located in ./users/reference/

ROI_LIST_Abnormal_Detection = [
        (144, 91, 172, 631),     # ROI1: 左側の範囲
        (1226, 92, 1251, 630)    # ROI2: 右側の範囲
    ] # Where to search for comparing abnormal and normal frames
