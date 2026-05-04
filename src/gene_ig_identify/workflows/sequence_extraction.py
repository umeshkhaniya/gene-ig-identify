"""Sequence extraction workflow."""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path

from ..io.tables import load_table, normalize_domain_table
from ..logging_utils import get_logger
from ..paths import get_path, resolve_path

LOGGER = get_logger(__name__)


def process_sequence_resids(input_table: Path, sequence_dir: Path):
    all_sequences = {}
    input_data = normalize_domain_table(load_table(input_table))
    for _, row in input_data.iterrows():
        pdb = row["pdb"].upper()
        chain = row["chainid"]
        begin, end = row["igdomain_res_range"].split("_")
        seq_file = sequence_dir / f"{pdb}_sequence.pkl.gz"
        with gzip.open(seq_file, "rb") as handle:
            seq_info = pickle.load(handle)
        res_list = []
        in_range = False
        pdb_chain_info = f"{pdb}_{chain}_{begin}_{end}"
        for resid_dict in seq_info[f"{pdb}_{chain}"]:
            resi = str(resid_dict["resi"])
            if resi == str(begin):
                in_range = True
            if in_range:
                res_list.append((resid_dict["name"], resi))
            if resi == str(end):
                break
        all_sequences[pdb_chain_info] = res_list
    return all_sequences


def run(config, input_table: Path, output_file: Path, sequence_dir: str | None = None) -> None:
    active_sequence_dir = resolve_path(config, sequence_dir) if sequence_dir else get_path(config, "sequence_dir")
    sequences = process_sequence_resids(input_table, active_sequence_dir)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_file, "wb") as handle:
        pickle.dump(sequences, handle)
    LOGGER.info("Saved %s extracted sequence entries to %s", len(sequences), output_file)

