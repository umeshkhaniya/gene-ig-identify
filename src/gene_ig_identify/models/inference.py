"""Model loading and inference helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from ..constants import EXPECTED_EDGE_FEATURES, EXPECTED_NODE_FEATURES
from ..io.artifacts import load_json
from ..labels import LABEL_MAPPING, REVERSE_LABEL_MAPPING
from .gine import GraphClassifier


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_model(model_dir: str | Path, device: torch.device):
    model_dir = Path(model_dir)
    best_params = load_json(model_dir / "best_hyperparameters.json")
    model_config = load_json(model_dir / "model_config.json")
    if int(model_config.get("num_classes", len(LABEL_MAPPING))) != len(LABEL_MAPPING):
        raise ValueError(
            f"Model config at {model_dir / 'model_config.json'} is incompatible with the stable "
            f"{len(LABEL_MAPPING)}-class label mapping."
        )
    if int(model_config.get("node_features", EXPECTED_NODE_FEATURES)) != EXPECTED_NODE_FEATURES:
        raise ValueError(
            f"Model config at {model_dir / 'model_config.json'} expects "
            f"{model_config.get('node_features')} node features, but the package expects "
            f"{EXPECTED_NODE_FEATURES}."
        )
    if int(model_config.get("edge_in_channels_featutes", EXPECTED_EDGE_FEATURES)) != EXPECTED_EDGE_FEATURES:
        raise ValueError(
            f"Model config at {model_dir / 'model_config.json'} expects "
            f"{model_config.get('edge_in_channels_featutes')} edge features, but the package expects "
            f"{EXPECTED_EDGE_FEATURES}."
        )
    model = GraphClassifier(
        in_channels=EXPECTED_NODE_FEATURES,
        edge_in_channels=EXPECTED_EDGE_FEATURES,
        hidden_dim=best_params["hidden_dim"],
        num_classes=len(REVERSE_LABEL_MAPPING),
        num_layers=best_params["num_layers"],
        dropout_rate=best_params["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(model_dir / "best_graph_model.pth", map_location=device))
    model.eval()
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
