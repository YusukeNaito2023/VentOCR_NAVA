ENGINE_SETTINGS_SCHEMA = {
    "initial_ocr": {
        "skip_keys": {
            "type": "multiselect",
            "choices": ["Cdyn", "EdiPeak", "Triggered", "VTe"],
            "default": [],
            "label": "ROI Keys to be skipped"
        },
        "similarity_thresh": {
            "type": "float",
            "default": 0.9,
            "label": "Triggered Threshold for Breath Mark"
        },
        "sampling_size": {
            "type": "int",
            "default": 10,
            "label": "Sampling Interval (Frames)"
        },
        "use_gpu": {
            "type": "bool",
            "default": True,
            "label": "Enable GPU Processing"
        }
    },
    "post_ocr": {
        "imputation_col": {
            "type": "multiselect",
            "choices": ["VTe", "EdiPeak", "VTi"],
            "default": ["VTe", "EdiPeak"],
            "label": "Imputation Columns"
        },
        "smoothing": {
            "type": "bool",
            "default": False,
            "label": "Apply Smoothing to Waveform"
        },
        "imputation": {
            "type": "bool",
            "default": False,
            "label": "Apply Interpolation"
        }
    }
}
