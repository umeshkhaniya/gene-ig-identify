"""Helpers for model and prediction artifacts."""

from __future__ import annotations

from pathlib import Path
import json


def ensure_artifact_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_json(data: dict, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    return output_path


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_torch(obj: object, path: str | Path) -> Path:
    import torch

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, output_path)
    return output_path


def load_torch(path: str | Path):
    import torch

    return torch.load(path, weights_only=False)
