"""Centralized stable label mapping."""

LABEL_MAPPING = {
    "IgV": 0,
    "IgC1": 1,
    "IgC2": 2,
    "IgI": 3,
    "Cadherin": 4,
    "IgFN3": 5,
    "Lamin": 6,
    "CD19": 7,
}

REVERSE_LABEL_MAPPING = {value: key for key, value in LABEL_MAPPING.items()}

