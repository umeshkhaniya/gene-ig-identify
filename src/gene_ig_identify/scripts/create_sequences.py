"""Simple per-domain sequence extraction script."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="create_sequences.py",
        description="Create the per-domain sequences pickle used by ESM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", help="Optional YAML config file.")
    parser.add_argument(
        "--input-table",
        required=True,
        help="Input Excel/CSV/TSV table with pdbid_chain and igdomain_res_range.",
    )
    parser.add_argument(
        "--sequence-dir",
        default="input/sequence_file",
        help="Folder with <PDB>_sequence.pkl.gz files created by the iCn3D step.",
    )
    parser.add_argument(
        "--output-file",
        default="output/sequences.pkl.gz",
        help="Where to write extracted per-domain sequences.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    from gene_ig_identify.config import load_config
    from gene_ig_identify.logging_utils import configure_logging
    from gene_ig_identify.paths import resolve_path
    from gene_ig_identify.workflows import sequence_extraction

    config = load_config(args.config)
    configure_logging(config.runtime.get("log_level", "INFO"))
    sequence_extraction.run(
        config,
        input_table=resolve_path(config, args.input_table),
        sequence_dir=args.sequence_dir,
        output_file=resolve_path(config, args.output_file),
    )


if __name__ == "__main__":
    main()

