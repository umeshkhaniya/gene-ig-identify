"""Simple ESM-2 embedding creation script."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="create_esm_embeddings.py",
        description="Create an HDF5 file of residue-level ESM-2 embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", help="Optional YAML config file.")
    parser.add_argument(
        "--input-file",
        default="output/sequences.pkl.gz",
        help="Per-domain sequences pickle created by create_sequences.py.",
    )
    parser.add_argument(
        "--output-file",
        default="output/esm_embeddings.h5",
        help="Where to write ESM embeddings in HDF5 format.",
    )
    parser.add_argument(
        "--model-name",
        default="esm2_t33_650M_UR50D",
        help="ESM-2 model name understood by fair-esm.",
    )
    parser.add_argument("--device", help="Device to use, for example cpu, cuda, or auto.")
    parser.add_argument("--cache-dir", help="Optional model cache directory.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    from gene_ig_identify.config import load_config
    from gene_ig_identify.logging_utils import configure_logging
    from gene_ig_identify.paths import resolve_path
    from gene_ig_identify.integrations import esm_embeddings

    config = load_config(args.config)
    configure_logging(config.runtime.get("log_level", "INFO"))
    esm_embeddings.run(
        config,
        input_file=resolve_path(config, args.input_file),
        output_file=resolve_path(config, args.output_file),
        model_name=args.model_name,
        device=args.device,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()

