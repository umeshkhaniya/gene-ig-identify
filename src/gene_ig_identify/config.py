"""Configuration loading from defaults, environment, and CLI overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - dependency check
    raise ModuleNotFoundError(
        "PyYAML is required to load configuration files. Install the package dependencies first, "
        "for example with: pip install -e ."
    ) from exc


@dataclass
class AppConfig:
    raw: dict[str, Any]
    config_path: Path
    project_root: Path

    @property
    def runtime(self) -> dict[str, Any]:
        return self.raw.setdefault("runtime", {})

    @property
    def paths(self) -> dict[str, Any]:
        return self.raw.setdefault("paths", {})

    @property
    def executables(self) -> dict[str, Any]:
        return self.raw.setdefault("executables", {})

    @property
    def model(self) -> dict[str, Any]:
        return self.raw.setdefault("model", {})


def _deep_update(target: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def load_config(config_path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> AppConfig:
    project_root = Path(__file__).resolve().parents[2]
    default_path = project_root / "config" / "default.yaml"
    active_config_path = Path(
        config_path
        or os.environ.get("GENE_IG_IDENTIFY_CONFIG", default_path)
    )
    base = _load_yaml(default_path)
    if active_config_path != default_path:
        _deep_update(base, _load_yaml(active_config_path))

    env_map = {
        "GENE_IG_IDENTIFY_INPUT_DIR": ("paths", "input_dir"),
        "GENE_IG_IDENTIFY_OUTPUT_DIR": ("paths", "output_dir"),
        "GENE_IG_IDENTIFY_ARTIFACTS_DIR": ("paths", "artifacts_dir"),
        "GENE_IG_IDENTIFY_MODELS_DIR": ("paths", "models_dir"),
        "GENE_IG_IDENTIFY_ESM_CACHE": ("paths", "esm_cache_dir"),
        "GENE_IG_IDENTIFY_DEVICE": ("runtime", "device"),
        "GENE_IG_IDENTIFY_NODE": ("executables", "node"),
    }
    for env_key, path_keys in env_map.items():
        value = os.environ.get(env_key)
        if value:
            cursor = base
            for key in path_keys[:-1]:
                cursor = cursor.setdefault(key, {})
            cursor[path_keys[-1]] = value

    if overrides:
        _deep_update(base, overrides)

    return AppConfig(raw=base, config_path=active_config_path, project_root=project_root)
