"""Residue feature generation workflow."""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path

from ..features.structure_geometry import ResidueAnalyzer
from ..io.tables import load_table, normalize_domain_table
from ..logging_utils import get_logger
from ..paths import ensure_dir, get_path, resolve_path

LOGGER = get_logger(__name__)


def process_pdb_files(input_excel: Path, input_dir: Path, pdb_subdir: str, structure_subdir: str, cutoff_distance: int) -> None:
    if not input_excel.exists():
        raise FileNotFoundError(f"Input table not found: {input_excel}")
    input_data = normalize_domain_table(load_table(input_excel))
    pdb_path = input_dir / pdb_subdir
    if not pdb_path.exists():
        raise FileNotFoundError(
            f"PDB input folder not found: {pdb_path}. Place PDB/CIF files in this folder before running "
            f"`gene-ig-identify features structures`."
        )
    structure_features_path = ensure_dir(input_dir / structure_subdir)
    for _, row in input_data.iterrows():
        pdb_chain = row["pdbid_chain"]
        res_range = row["igdomain_res_range"]
        LOGGER.info("Processing %s:%s", pdb_chain, res_range)
        domain_begin, domain_end = res_range.split("_")
        pdb = row["pdb"].upper()
        chain = row["chainid"]
        if len(chain) != 1:
            continue
        pdb_file = pdb_path / pdb
        structure_features_file = structure_features_path / f"{pdb}_{chain}_{domain_begin}_{domain_end}_structure.pkl.gz"
        if structure_features_file.exists():
            continue
        pdb_exists = any((pdb_file.with_suffix(ext)).exists() for ext in [".pdb", ".cif"])
        if not pdb_exists:
            raise FileNotFoundError(
                f"Expected PDB/CIF file for {pdb} in {pdb_path}. Place the required structure files in "
                f"that folder before running feature generation."
            )
        try:
            analyzer = ResidueAnalyzer(str(pdb_file), chain)
            residues_features = analyzer.analyze_residues(domain_begin, domain_end, cutoff_distance)
            if residues_features:
                with gzip.open(structure_features_file, "wb") as handle:
                    pickle.dump(residues_features, handle)
        except Exception as exc:
            LOGGER.exception("Error processing %s: %s", pdb_chain, exc)


def run(config, input_table: Path, input_dir: str | None, pdb_subdir: str, structure_subdir: str, cutoff_distance: int) -> None:
    active_input_dir = resolve_path(config, input_dir) if input_dir else get_path(config, "input_dir")
    process_pdb_files(input_table, active_input_dir, pdb_subdir, structure_subdir, cutoff_distance)
