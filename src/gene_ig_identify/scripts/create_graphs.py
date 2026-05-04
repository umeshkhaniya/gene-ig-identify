"""Simple graph-building script.

This module backs both:

    python src/create_graphs.py ...
    gene-ig-create-graphs ...
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="create_graphs.py",
        description="Create graph files from an input domain table and prepared feature files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", help="Optional YAML config file.")
    parser.add_argument(
        "--input-table",
        required=True,
        help="Input Excel/CSV/TSV table with pdbid_chain and igdomain_res_range. ig_type is optional for prediction.",
    )
    parser.add_argument(
        "--embeddings-file",
        required=True,
        help="HDF5 ESM embeddings file created for the same domains.",
    )
    parser.add_argument(
        "--graphs-output",
        default="results/graphs/all_graphs.pt",
        help="Where to write the list of graph objects.",
    )
    parser.add_argument(
        "--graph-lookup-output",
        default="results/graphs/graph_lookup.pt",
        help="Where to write the graph lookup dictionary.",
    )
    parser.add_argument(
        "--pdb-dir",
        help="Folder with <PDB>.pdb or <PDB>.cif files. Defaults to paths.pdb_dir from config.",
    )
    parser.add_argument(
        "--icn3dss-dir",
        help="Folder with <PDB>_icn3dss.pkl.gz files. Defaults to paths.icn3dss_dir from config.",
    )
    parser.add_argument(
        "--structure-features-dir",
        help="Folder with <PDB>_<CHAIN>_<BEGIN>_<END>_structure.pkl.gz files. Defaults to paths.structure_features_dir from config.",
    )
    parser.add_argument(
        "--icn3d-interactions-dir",
        help="Folder with <PDB>_<CHAIN>_icn3dinteraction.json files. Defaults to paths.icn3d_interactions_dir from config.",
    )
    return parser


def _path_arg(config, value: str | None, config_key: str) -> Path:
    from gene_ig_identify.paths import get_path, resolve_path

    if value:
        return resolve_path(config, value)
    return get_path(config, config_key)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    from gene_ig_identify.config import load_config
    from gene_ig_identify.logging_utils import configure_logging
    from gene_ig_identify.paths import resolve_path

    config = load_config(args.config)
    configure_logging(config.runtime.get("log_level", "INFO"))

    from gene_ig_identify.workflows import graph_building

    graph_building.run(
        config,
        input_table=resolve_path(config, args.input_table),
        pdb_dir=_path_arg(config, args.pdb_dir, "pdb_dir"),
        icn3dss_dir=_path_arg(config, args.icn3dss_dir, "icn3dss_dir"),
        structure_features_dir=_path_arg(config, args.structure_features_dir, "structure_features_dir"),
        icn3d_interactions_dir=_path_arg(config, args.icn3d_interactions_dir, "icn3d_interactions_dir"),
        embeddings_file=resolve_path(config, args.embeddings_file),
        graphs_output=resolve_path(config, args.graphs_output),
        graph_lookup_output=resolve_path(config, args.graph_lookup_output),
    )


if __name__ == "__main__":
    main()
