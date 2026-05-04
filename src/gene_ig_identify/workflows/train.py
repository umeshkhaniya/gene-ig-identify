"""Training workflow with stratified 5-fold hyperparameter tuning."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from ..constants import EXPECTED_EDGE_FEATURES, EXPECTED_NODE_FEATURES
from ..io.artifacts import ensure_artifact_dir, load_torch, save_json, save_torch
from ..labels import LABEL_MAPPING
from ..logging_utils import get_logger
from ..models.dataset_split import set_seed
from ..models.gine import GraphClassifier

LOGGER = get_logger(__name__)

NUM_CLASSES = len(LABEL_MAPPING)
CV_FOLDS = 5
RANDOM_STATE = 42


def _graph_labels(graphs) -> list[int]:
    return [int(data.y.item()) for data in graphs]


def _validate_labeled_graphs(graphs, source: Path) -> None:
    if not isinstance(graphs, list) or not graphs:
        raise ValueError(f"Expected a non-empty list of graphs in {source}.")
    missing_labels = [idx for idx, data in enumerate(graphs) if not hasattr(data, "y") or data.y is None]
    if missing_labels:
        raise ValueError("Training requires labeled graphs. Add an ig_type column before graph creation.")
    invalid_labels = sorted({label for label in _graph_labels(graphs) if label < 0 or label >= NUM_CLASSES})
    if invalid_labels:
        raise ValueError(f"Graph labels outside stable {NUM_CLASSES}-class mapping: {invalid_labels}")


def _class_weights(graphs, device: torch.device) -> torch.Tensor:
    labels = torch.tensor(_graph_labels(graphs), dtype=torch.long)
    class_counts = torch.bincount(labels, minlength=NUM_CLASSES).float()
    class_weights = torch.zeros(NUM_CLASSES, dtype=torch.float)
    present = class_counts > 0
    class_weights[present] = 1.0 / torch.log1p(class_counts[present])
    class_weights = class_weights / class_weights[present].sum()
    return class_weights.to(device)


def _make_model(params: dict, device: torch.device) -> GraphClassifier:
    return GraphClassifier(
        in_channels=EXPECTED_NODE_FEATURES,
        edge_in_channels=EXPECTED_EDGE_FEATURES,
        hidden_dim=params["hidden_dim"],
        num_classes=NUM_CLASSES,
        num_layers=params["num_layers"],
        dropout_rate=params["dropout"],
    ).to(device)


def _train_epoch(model, loader, optimizer, loss_fn, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_graphs = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        batch_size = int(data.y.size(0))
        total_loss += float(loss.item()) * batch_size
        total_graphs += batch_size
    return total_loss / total_graphs if total_graphs else 0.0


def _evaluate(model, loader, loss_fn, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = loss_fn(out, data.y)
            batch_size = int(data.y.size(0))
            total_loss += float(loss.item()) * batch_size
            preds = out.argmax(dim=1)
            correct += int((preds == data.y).sum().item())
            total += batch_size
    return (total_loss / total if total else 0.0, correct / total if total else 0.0)


def _validate_five_fold_possible(labels: list[int]) -> None:
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    present_counts = class_counts[class_counts > 0]
    if len(present_counts) < 2:
        raise ValueError("Training requires at least two classes.")
    min_count = int(present_counts.min())
    if min_count < CV_FOLDS:
        raise ValueError(
            f"Stratified {CV_FOLDS}-fold cross-validation requires at least {CV_FOLDS} graphs "
            f"for every class present in the training/validation split; smallest class has {min_count}."
        )


def _split_train_val_test(graphs: list) -> tuple[list, list, list[int]]:
    labels = _graph_labels(graphs)
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    present_counts = class_counts[class_counts > 0]
    if len(present_counts) < 2:
        raise ValueError("Training requires at least two classes.")
    min_count = int(present_counts.min())
    if min_count < 2:
        raise ValueError("The held-out test split requires at least two graphs for every class present.")
    present_classes = len(present_counts)
    test_count = max(present_classes, int(np.ceil(0.2 * len(graphs))))
    if len(graphs) - test_count < present_classes:
        raise ValueError("Not enough graphs to create stratified train/test splits.")
    train_val_idx, test_idx = train_test_split(
        range(len(graphs)),
        test_size=test_count,
        stratify=labels,
        random_state=RANDOM_STATE,
    )
    train_val_graphs = [graphs[i] for i in train_val_idx]
    test_graphs = [graphs[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    _validate_five_fold_possible(_graph_labels(train_val_graphs))
    return train_val_graphs, test_graphs, test_labels


def _final_train_validation_split(graphs: list, labels: list[int]) -> tuple[list, list]:
    present_classes = len(set(labels))
    validation_count = max(present_classes, int(np.ceil(0.1 * len(graphs))))
    train_idx, val_idx = train_test_split(
        range(len(graphs)),
        test_size=validation_count,
        stratify=labels,
        random_state=RANDOM_STATE,
    )
    return [graphs[i] for i in train_idx], [graphs[i] for i in val_idx]


def run(config, graphs_file: Path, graph_lookup_file: Path, output_dir: Path, epochs: int, trials: int) -> None:
    set_seed(RANDOM_STATE)
    output_dir = ensure_artifact_dir(output_dir)

    all_graphs = load_torch(graphs_file)
    _validate_labeled_graphs(all_graphs, graphs_file)

    graph_lookup = load_torch(graph_lookup_file)
    graph_list = list(graph_lookup.values())
    _validate_labeled_graphs(graph_list, graph_lookup_file)

    train_val_graphs, test_graphs, test_labels = _split_train_val_test(graph_list)
    save_torch(test_graphs, output_dir / "test_graphs.pt")
    save_torch(test_labels, output_dir / "test_labels.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tuning_epochs = min(20, max(1, epochs))
    train_val_labels = _graph_labels(train_val_graphs)
    LOGGER.info(
        "Training with %s train/validation graphs, %s held-out test graphs, %s-fold CV, %s Optuna trials",
        len(train_val_graphs),
        len(test_graphs),
        CV_FOLDS,
        trials,
    )

    def objective(trial):
        params = {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "num_layers": trial.suggest_int("num_layers", 2, 4),
        }
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_losses = []
        fold_accuracies = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_val_graphs, train_val_labels), start=1):
            fold_train_graphs = [train_val_graphs[i] for i in train_idx]
            fold_val_graphs = [train_val_graphs[i] for i in val_idx]
            train_loader = DataLoader(fold_train_graphs, batch_size=params["batch_size"], shuffle=True)
            val_loader = DataLoader(fold_val_graphs, batch_size=params["batch_size"], shuffle=False)
            model = _make_model(params, device)
            optimizer = optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
            loss_fn = torch.nn.CrossEntropyLoss(weight=_class_weights(fold_train_graphs, device))
            for _ in range(tuning_epochs):
                train_loss = _train_epoch(model, train_loader, optimizer, loss_fn, device)
                scheduler.step(train_loss)
            val_loss, val_acc = _evaluate(model, val_loader, loss_fn, device)
            fold_losses.append(val_loss)
            fold_accuracies.append(val_acc)
            LOGGER.info(
                "Trial %s fold %s/%s val_loss %.4f val_acc %.4f",
                trial.number,
                fold_idx,
                CV_FOLDS,
                val_loss,
                val_acc,
            )
        trial.set_user_attr("mean_cv_accuracy", float(np.mean(fold_accuracies)))
        return float(np.mean(fold_losses))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trials)
    best_params = study.best_params

    final_train_graphs, final_val_graphs = _final_train_validation_split(train_val_graphs, train_val_labels)
    batch_size = best_params["batch_size"]
    train_loader = DataLoader(final_train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(final_val_graphs, batch_size=batch_size, shuffle=False)
    model = _make_model(best_params, device)
    optimizer = optim.AdamW(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    loss_fn = torch.nn.CrossEntropyLoss(weight=_class_weights(final_train_graphs, device))

    train_losses = []
    val_accuracies = []
    for epoch in range(epochs):
        train_loss = _train_epoch(model, train_loader, optimizer, loss_fn, device)
        scheduler.step(train_loss)
        _, val_acc = _evaluate(model, val_loader, loss_fn, device)
        train_losses.append(train_loss)
        val_accuracies.append(val_acc)
        LOGGER.info("Epoch %s loss %.4f val_acc %.4f", epoch + 1, train_loss, val_acc)

    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    _, test_acc = _evaluate(model, test_loader, loss_fn, device)
    LOGGER.info("Final held-out test accuracy %.4f", test_acc)

    torch.save(model.state_dict(), output_dir / "best_graph_model.pth")
    save_json(best_params, output_dir / "best_hyperparameters.json")
    save_json(
        {
            "best_params": best_params,
            "node_features": EXPECTED_NODE_FEATURES,
            "edge_in_channels_featutes": EXPECTED_EDGE_FEATURES,
            "num_classes": NUM_CLASSES,
            "cv_folds": CV_FOLDS,
            "test_accuracy": test_acc,
        },
        output_dir / "model_config.json",
    )
    save_json(
        {
            "best_value": float(study.best_value),
            "best_trial": int(study.best_trial.number),
            "best_mean_cv_accuracy": float(study.best_trial.user_attrs.get("mean_cv_accuracy", 0.0)),
            "cv_folds": CV_FOLDS,
            "tuning_epochs_per_fold": tuning_epochs,
            "final_test_accuracy": test_acc,
        },
        output_dir / "cross_validation_summary.json",
    )

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(range(1, epochs + 1), train_losses, label="Training Loss", marker="o", color="blue")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax2 = ax1.twinx()
    ax2.plot(range(1, epochs + 1), val_accuracies, label="Validation Accuracy", marker="o", color="orange")
    ax2.set_ylabel("Validation Accuracy", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")
    fig.suptitle("Training Loss and Validation Accuracy per Epoch")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_accuracy_plot_hybrid.png")
    plt.close(fig)
