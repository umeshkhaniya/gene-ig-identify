"""Model loading and inference helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from ..constants import EXPECTED_EDGE_FEATURES, EXPECTED_NODE_FEATURES
from ..io.artifacts import load_json
from ..labels import (
    LABEL_MAPPING,
    REVERSE_LABEL_MAPPING,
    build_label_mapping,
    build_reverse_label_mapping,
)
from .gine import GraphClassifier


@dataclass(frozen=True)
class ModelLabelSpace:
    label_mapping: dict[str, int]
    reverse_label_mapping: dict[int, str]
    labels: list[str]
    experiment: str
    name: str
    is_legacy: bool = False

    @property
    def num_classes(self) -> int:
        return len(self.labels)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _labels_from_model_config(model_config: dict, source: Path) -> list[str]:
    labels = model_config["labels"]
    if isinstance(labels, (str, bytes)) or not isinstance(labels, list):
        raise ValueError(f"Model config at {source} has invalid labels metadata.")
    normalized = []
    for label in labels:
        text = str(label).strip()
        if not text:
            raise ValueError(f"Model config at {source} contains an empty label name.")
        normalized.append(text)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Model config at {source} contains duplicate label names.")
    return normalized


def _label_mapping_from_model_config(model_config: dict, labels: list[str], source: Path) -> dict[str, int]:
    raw_mapping = model_config["label_mapping"]
    if not isinstance(raw_mapping, dict):
        raise ValueError(f"Model config at {source} has invalid label_mapping metadata.")
    try:
        label_mapping = {
            str(label): int(index)
            for label, index in raw_mapping.items()
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Model config at {source} has non-integer label_mapping values.") from exc

    expected_mapping = build_label_mapping(labels)
    if label_mapping != expected_mapping:
        raise ValueError(f"Model config at {source} label_mapping does not match labels order.")
    return label_mapping


def model_label_space_from_config(model_config: dict, source: str | Path = "model_config.json") -> ModelLabelSpace:
    source_path = Path(source)
    has_labels = "labels" in model_config
    has_label_mapping = "label_mapping" in model_config
    if has_labels != has_label_mapping:
        raise ValueError(f"Model config at {source_path} must contain both labels and label_mapping.")

    if not has_labels:
        legacy_num_classes = int(model_config.get("num_classes", len(LABEL_MAPPING)))
        if legacy_num_classes != len(LABEL_MAPPING):
            raise ValueError(
                f"Model config at {source_path} is missing labels metadata and is incompatible with the "
                f"legacy {len(LABEL_MAPPING)}-class EXP00 label mapping."
            )
        return ModelLabelSpace(
            label_mapping=dict(LABEL_MAPPING),
            reverse_label_mapping=dict(REVERSE_LABEL_MAPPING),
            labels=[REVERSE_LABEL_MAPPING[index] for index in range(len(REVERSE_LABEL_MAPPING))],
            experiment=str(model_config.get("experiment", "EXP00")),
            name=str(model_config.get("name", "8class_baseline")),
            is_legacy=True,
        )

    labels = _labels_from_model_config(model_config, source_path)
    label_mapping = _label_mapping_from_model_config(model_config, labels, source_path)
    num_classes = int(model_config.get("num_classes", len(labels)))
    if num_classes != len(labels):
        raise ValueError(
            f"Model config at {source_path} declares {num_classes} classes but lists {len(labels)} labels."
        )
    return ModelLabelSpace(
        label_mapping=label_mapping,
        reverse_label_mapping=build_reverse_label_mapping(label_mapping),
        labels=labels,
        experiment=str(model_config.get("experiment", "")),
        name=str(model_config.get("name", "")),
    )


def load_model_label_space(model_dir: str | Path) -> ModelLabelSpace:
    model_dir = Path(model_dir)
    model_config_path = model_dir / "model_config.json"
    return model_label_space_from_config(load_json(model_config_path), model_config_path)


def load_model_artifacts(model_dir: str | Path, device: torch.device):
    model_dir = Path(model_dir)
    best_params = load_json(model_dir / "best_hyperparameters.json")
    model_config_path = model_dir / "model_config.json"
    model_config = load_json(model_config_path)
    label_space = model_label_space_from_config(model_config, model_config_path)
    if int(model_config.get("node_features", EXPECTED_NODE_FEATURES)) != EXPECTED_NODE_FEATURES:
        raise ValueError(
            f"Model config at {model_config_path} expects "
            f"{model_config.get('node_features')} node features, but the package expects "
            f"{EXPECTED_NODE_FEATURES}."
        )
    edge_features = model_config.get(
        "edge_in_channels_features",
        model_config.get("edge_in_channels_featutes", EXPECTED_EDGE_FEATURES),
    )
    if int(edge_features) != EXPECTED_EDGE_FEATURES:
        raise ValueError(
            f"Model config at {model_config_path} expects "
            f"{edge_features} edge features, but the package expects "
            f"{EXPECTED_EDGE_FEATURES}."
        )
    model = GraphClassifier(
        in_channels=EXPECTED_NODE_FEATURES,
        edge_in_channels=EXPECTED_EDGE_FEATURES,
        hidden_dim=best_params["hidden_dim"],
        num_classes=label_space.num_classes,
        num_layers=best_params["num_layers"],
        dropout_rate=best_params["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(model_dir / "best_graph_model.pth", map_location=device))
    model.eval()
    return model, best_params, label_space


def load_model(model_dir: str | Path, device: torch.device):
    model, best_params, _ = load_model_artifacts(model_dir, device)
    return model, best_params


def predict_graphs(graphs, model, batch_size: int, device: torch.device):
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    all_probs = []
    all_preds = []
    all_labels = []
    all_graph_names = []
    with torch.no_grad():
        for data in loader:
            batch_names = list(data.unique_name_file)
            data = data.to(device)
            out = model(data)
            probs = F.softmax(out, dim=1)
            preds = probs.argmax(dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_graph_names.extend(batch_names)
            if hasattr(data, "y") and data.y is not None:
                all_labels.extend(data.y.cpu().numpy())
    return np.array(all_preds), np.array(all_probs), np.array(all_labels), all_graph_names
