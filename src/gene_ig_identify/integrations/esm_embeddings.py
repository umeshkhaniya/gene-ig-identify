"""ESM embeddings workflow."""

from __future__ import annotations

import gzip
import os
import pickle
from pathlib import Path

import h5py
import numpy as np
import torch

from ..logging_utils import get_logger
from ..paths import ensure_dir, get_path, resolve_path

LOGGER = get_logger(__name__)


def _load_esm():
    try:
        import esm
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The ESM-2 embedding step requires fair-esm. Install project dependencies with "
            "`python -m pip install -e .` or install it directly with "
            "`python -m pip install fair-esm`."
        ) from exc
    return esm


def load_data(file_path):
    if not Path(file_path).exists():
        raise FileNotFoundError(
            f"Sequence input file not found: {file_path}. Create it first with `python src/create_sequences.py`."
        )
    with gzip.open(file_path, "rb") as handle:
        return pickle.load(handle)


def save_embeddings_hdf5(embeddings, output_file):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5f:
        for protein_id, residue_dict in embeddings.items():
            group = h5f.create_group(protein_id)
            for res_id, emb_tensor in residue_dict.items():
                emb_array = emb_tensor.cpu().numpy() if isinstance(emb_tensor, torch.Tensor) else np.asarray(emb_tensor)
                group.create_dataset(res_id, data=emb_array, compression="gzip", compression_opts=4)


def get_esmfold_embeddings(data, model_name, device):
    esm = _load_esm()
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()
    result = {}
    for key, residues in data.items():
        sequence = "".join(res[0] for res in residues)
        residue_ids = [f"{res[1]}_{res[0]}" for res in residues]
        batch = [(key, sequence)]
        _, _, batch_tokens = batch_converter(batch)
        batch_tokens = batch_tokens.to(device)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        with torch.no_grad():
            output = model(batch_tokens, repr_layers=[model.num_layers], return_contacts=False)
        token_representations = output["representations"][model.num_layers]
        embeddings = token_representations[0, 1:batch_lens[0] - 1].cpu()
        if len(embeddings) != len(residue_ids):
            raise ValueError(f"Embedding/residue mismatch for {key}")
        result[key] = {res_id: emb for res_id, emb in zip(residue_ids, embeddings)}
    return result


def run(config, input_file: Path, output_file: Path, model_name: str, device: str | None = None, cache_dir: str | None = None) -> None:
    active_cache_dir = cache_dir or config.paths.get("esm_cache_dir")
    if active_cache_dir:
        cache_path = resolve_path(config, active_cache_dir)
        ensure_dir(cache_path)
        os.environ["TORCH_HOME"] = str(cache_path)
    requested_device = (device or config.runtime.get("device", "auto")).lower()
    if requested_device == "auto":
        actual_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        actual_device = torch.device(requested_device)
    LOGGER.info("Loading sequences from %s", input_file)
    input_data = load_data(input_file)
    LOGGER.info("Loaded %s protein entries", len(input_data))
    embeddings = get_esmfold_embeddings(input_data, model_name, actual_device)
    save_embeddings_hdf5(embeddings, output_file)
    LOGGER.info("Saved embeddings to %s", output_file)
