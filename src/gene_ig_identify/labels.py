"""Utilities for creating label mappings."""

from collections.abc import Mapping, Sequence
from typing import Any


def build_label_mapping(labels: Sequence[str]) -> dict[str, int]:
    """Create a label to integer mapping."""
    return {label: index for index, label in enumerate(labels)}


def build_reverse_label_mapping(
    label_mapping: dict[str, int],
) -> dict[int, str]:
    """Create an integer to label mapping."""
    return {value: key for key, value in label_mapping.items()}


# Backward-compatible EXP00 defaults used by existing pipeline code.
DEFAULT_LABELS = [
    "IgV",
    "IgC1",
    "IgC2",
    "IgI",
    "Cadherin",
    "IgFN3",
    "Lamin",
    "CD19",
]


def labels_from_config(config: Any) -> list[str]:
    """Return the ordered experiment labels from a loaded config."""
    raw = getattr(config, "raw", config)
    if not isinstance(raw, Mapping):
        raise TypeError("Expected an AppConfig or config mapping with a labels entry.")

    configured_labels = raw.get("labels", DEFAULT_LABELS)
    if isinstance(configured_labels, (str, bytes)) or not isinstance(configured_labels, Sequence):
        raise ValueError("Config labels must be a list of label names.")

    normalized_labels = []
    for label in configured_labels:
        text = str(label).strip()
        if not text:
            raise ValueError("Config labels must not contain empty values.")
        normalized_labels.append(text)

    duplicates = []
    seen = set()
    for label in normalized_labels:
        if label in seen and label not in duplicates:
            duplicates.append(label)
        seen.add(label)
    if duplicates:
        raise ValueError(f"Config labels must be unique. Duplicates: {', '.join(duplicates)}")

    return normalized_labels


def label_mapping_from_config(config: Any) -> dict[str, int]:
    """Create an ordered label mapping from a loaded config."""
    return build_label_mapping(labels_from_config(config))


def reverse_label_mapping_from_config(config: Any) -> dict[int, str]:
    """Create a reverse label mapping from a loaded config."""
    return build_reverse_label_mapping(label_mapping_from_config(config))


LABEL_MAPPING = build_label_mapping(DEFAULT_LABELS)

REVERSE_LABEL_MAPPING = build_reverse_label_mapping(LABEL_MAPPING)
