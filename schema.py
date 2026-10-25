ENGINE_SETTINGS_SCHEMA = {
    "initial_ocr": {
        "skip_keys": {
            "type": "multiselect",
            "choices": ["Cdyn", "EdiPeak", "VTe", "Ppeak", "VTi", "IE_Ratio"],
            "default": [],
            "label": "ROI Keys to be skipped"
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
        },
        "trim_to_breath": {
            "type": "bool",
            "default": True,
            "label": "Trim Waveform to Breath (inspiration onset)"
        }
    }
}
