"""Portable path handling."""

from __future__ import annotations

from pathlib import Path
import os

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

