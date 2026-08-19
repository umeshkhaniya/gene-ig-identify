"""CLI entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .labels import LABEL_MAPPING
from .logging_utils import configure_logging
from .paths import (
    get_experiment_metrics_dir,
    get_experiment_models_dir,
    get_experiment_predictions_dir,
    get_path,
    resolve_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gene-ig-identify")
    parser.add_argument("--config", help="Path to YAML config file.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    features_parser = subparsers.add_parser("features")
    features_sub = features_parser.add_subparsers(dest="features_command", required=True)

    icn3d_parser = features_sub.add_parser("icn3d")
    icn3d_parser.add_argument("--input-table", required=True)
    icn3d_parser.add_argument("--input-dir")
    icn3d_parser.add_argument("--node-executable")
    icn3d_parser.add_argument("--sequence-script")
    icn3d_parser.add_argument("--interaction-script")
    icn3d_parser.add_argument("--ss-script")

    structure_parser = features_sub.add_parser("structures")
    structure_parser.add_argument("--input-table", required=True)
    structure_parser.add_argument("--input-dir")
    structure_parser.add_argument("--pdb-subdir", default="pdb_files")
    structure_parser.add_argument("--structure-subdir", default="structure_features_residues")
    structure_parser.add_argument("--cutoff-distance", type=int, default=8)

    seq_parser = subparsers.add_parser("sequences")
    seq_sub = seq_parser.add_subparsers(dest="sequences_command", required=True)
    seq_extract = seq_sub.add_parser("extract")
    seq_extract.add_argument("--input-table", required=True)
    seq_extract.add_argument("--sequence-dir")
    seq_extract.add_argument("--output-file", required=True)

    emb_parser = subparsers.add_parser("embeddings")
    emb_sub = emb_parser.add_subparsers(dest="embeddings_command", required=True)
    emb_esm = emb_sub.add_parser("esm")
    emb_esm.add_argument("--input-file", required=True)
    emb_esm.add_argument("--output-file", required=True)
    emb_esm.add_argument("--model-name", default="esm2_t33_650M_UR50D")
    emb_esm.add_argument("--device")
    emb_esm.add_argument("--cache-dir")

    graphs_parser = subparsers.add_parser("graphs")
    graphs_sub = graphs_parser.add_subparsers(dest="graphs_command", required=True)
    graphs_build = graphs_sub.add_parser("build")
    graphs_build.add_argument("--input-table", required=True)
    graphs_build.add_argument("--pdb-dir")
    graphs_build.add_argument("--icn3dss-dir")
    graphs_build.add_argument("--structure-features-dir")
    graphs_build.add_argument("--icn3d-interactions-dir")
    graphs_build.add_argument("--embeddings-file", required=True)
    graphs_build.add_argument("--graphs-output", required=True)
    graphs_build.add_argument("--graph-lookup-output", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--graphs-file", required=True)
    train_parser.add_argument("--graph-lookup-file", required=True)
    train_parser.add_argument("--output-dir")
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--trials", type=int, default=30)

    predict_parser = subparsers.add_parser("predict")
    predict_sub = predict_parser.add_subparsers(dest="predict_command", required=True)
    predict_dataset = predict_sub.add_parser("dataset")
    predict_dataset.add_argument("--graphs-file", required=True)
    predict_dataset.add_argument("--excel-file", required=True)
    predict_dataset.add_argument("--model-dir")
    predict_dataset.add_argument("--output-dir")

    post_parser = subparsers.add_parser("postprocess")
    post_sub = post_parser.add_subparsers(dest="postprocess_command", required=True)
    post_merge = post_sub.add_parser("merge")
    post_merge.add_argument("--excel-file", required=True)
    post_merge.add_argument("--predictions-file", required=True)
    post_merge.add_argument("--uniprot-json", required=True)
    post_merge.add_argument("--output-file")

    post_arch = post_sub.add_parser("architecture")
    post_arch.add_argument("--input-file", required=True)
    post_arch.add_argument("--output-file", required=True)
    post_arch.add_argument("--skip-single-domain", action="store_true")

    config_parser = subparsers.add_parser("config")
    config_sub = config_parser.add_subparsers(dest="config_command", required=True)
    config_sub.add_parser("validate")

    labels_parser = subparsers.add_parser("labels")
    labels_sub = labels_parser.add_subparsers(dest="labels_command", required=True)
    labels_sub.add_parser("show")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    configure_logging(config.runtime.get("log_level", "INFO"))

    if args.command == "features" and args.features_command == "icn3d":
        from .integrations import icn3d

        icn3d.run_feature_collection(
            config,
            input_table=Path(args.input_table),
            input_dir=args.input_dir,
            node_executable=args.node_executable,
            sequence_script=args.sequence_script,
            interaction_script=args.interaction_script,
            ss_script=args.ss_script,
        )
    elif args.command == "features" and args.features_command == "structures":
        from .workflows import feature_generation

        feature_generation.run(
            config,
            input_table=Path(args.input_table),
            input_dir=args.input_dir,
            pdb_subdir=args.pdb_subdir,
            structure_subdir=args.structure_subdir,
            cutoff_distance=args.cutoff_distance,
        )
    elif args.command == "sequences" and args.sequences_command == "extract":
        from .workflows import sequence_extraction

        sequence_extraction.run(
            config,
            input_table=Path(args.input_table),
            output_file=Path(args.output_file),
            sequence_dir=args.sequence_dir,
        )
    elif args.command == "embeddings" and args.embeddings_command == "esm":
        from .integrations import esm_embeddings

        esm_embeddings.run(
            config,
            input_file=Path(args.input_file),
            output_file=Path(args.output_file),
            model_name=args.model_name,
            device=args.device,
            cache_dir=args.cache_dir,
        )
    elif args.command == "graphs" and args.graphs_command == "build":
        from .workflows import graph_building

        graph_building.run(
            config,
            input_table=Path(args.input_table),
            pdb_dir=resolve_path(config, args.pdb_dir) if args.pdb_dir else get_path(config, "pdb_dir"),
            icn3dss_dir=resolve_path(config, args.icn3dss_dir) if args.icn3dss_dir else get_path(config, "icn3dss_dir"),
            structure_features_dir=resolve_path(config, args.structure_features_dir) if args.structure_features_dir else get_path(config, "structure_features_dir"),
            icn3d_interactions_dir=resolve_path(config, args.icn3d_interactions_dir) if args.icn3d_interactions_dir else get_path(config, "icn3d_interactions_dir"),
            embeddings_file=Path(args.embeddings_file),
            graphs_output=Path(args.graphs_output),
            graph_lookup_output=Path(args.graph_lookup_output),
        )
    elif args.command == "train":
        from .workflows import train

        train.run(
            config,
            graphs_file=Path(args.graphs_file),
            graph_lookup_file=Path(args.graph_lookup_file),
            output_dir=resolve_path(config, args.output_dir) if args.output_dir else get_experiment_models_dir(config),
            epochs=args.epochs,
            trials=args.trials,
            metrics_dir=None if args.output_dir else get_experiment_metrics_dir(config),
        )
    elif args.command == "predict" and args.predict_command == "dataset":
        from .workflows import predict

        predict.run_excel_predictions(
            config,
            graphs_file=Path(args.graphs_file),
            excel_file=Path(args.excel_file),
            model_dir=resolve_path(config, args.model_dir) if args.model_dir else get_experiment_models_dir(config),
            output_dir=resolve_path(config, args.output_dir) if args.output_dir else get_experiment_predictions_dir(config),
        )
    elif args.command == "postprocess" and args.postprocess_command == "merge":
        from .workflows import postprocess

        postprocess.merge_predictions_with_annotations(
            config,
            excel_file=Path(args.excel_file),
            predictions_file=Path(args.predictions_file),
            uniprot_json=Path(args.uniprot_json),
            output_file=Path(args.output_file) if args.output_file else None,
        )
    elif args.command == "postprocess" and args.postprocess_command == "architecture":
        from .workflows import postprocess

        postprocess.create_chain_architecture_summary(
            input_file=Path(args.input_file),
            output_file=Path(args.output_file),
            skip_single_domain=args.skip_single_domain,
        )
    elif args.command == "config" and args.config_command == "validate":
        print(f"Loaded config: {config.config_path}")
        for key, value in config.paths.items():
            print(f"{key}: {value}")
    elif args.command == "labels" and args.labels_command == "show":
        for label, idx in LABEL_MAPPING.items():
            print(f"{label}: {idx}")
