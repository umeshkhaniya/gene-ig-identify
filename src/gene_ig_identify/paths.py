"""Portable path handling."""

from __future__ import annotations

from pathlib import Path
import os
import re

from .config import AppConfig


def resolve_path(config: AppConfig, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config.project_root / path).resolve()


def get_path(config: AppConfig, key: str) -> Path:
    value = config.paths.get(key)
    if value is None:
        raise KeyError(f"Missing configured path: {key}")
    return resolve_path(config, value)


def get_experiment_id(config: AppConfig) -> str:
    experiment = config.raw.get("experiment", {})
    experiment_id = str(experiment.get("id", "EXP00")).strip()
    if not experiment_id:
        raise ValueError("Missing experiment id in config.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", experiment_id):
        raise ValueError(f"Invalid experiment id for output paths: {experiment_id!r}")
    return experiment_id


def get_experiment_dir(config: AppConfig) -> Path:
    return get_path(config, "artifacts_dir") / "experiments" / get_experiment_id(config)


def get_experiment_models_dir(config: AppConfig) -> Path:
    return get_experiment_dir(config) / "models"


def get_experiment_metrics_dir(config: AppConfig) -> Path:
    return get_experiment_dir(config) / "metrics"


def get_experiment_predictions_dir(config: AppConfig) -> Path:
    return get_experiment_dir(config) / "predictions"


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def default_runtime_device(config: AppConfig) -> str:
    return str(config.runtime.get("device", "auto"))


def get_scratch_dir() -> Path | None:
    for env_name in ("SLURM_TMPDIR", "SCRATCH"):
        value = os.environ.get(env_name)
        if value:
            return Path(value)
    return None
