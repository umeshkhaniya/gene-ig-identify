"""Utilities for creating label mappings."""

from typing import Sequence


def build_label_mapping(labels: Sequence[str]) -> dict[str, int]:
    """Create a label to integer mapping."""
    return {label: index for index, label in enumerate(labels)}


def build_reverse_label_mapping(
    label_mapping: dict[str, int],
) -> dict[int, str]:
    """Create an integer to label mapping."""
    return {value: key for key, value in label_mapping.items()}


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

LABEL_MAPPING = build_label_mapping(DEFAULT_LABELS)

REVERSE_LABEL_MAPPING = build_reverse_label_mapping(LABEL_MAPPING)