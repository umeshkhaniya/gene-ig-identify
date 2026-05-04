"""ICN3D and Node integration."""

from __future__ import annotations

from pathlib import Path
import gzip
import json
import pickle
import shutil
import subprocess

import pandas as pd

from ..io.tables import load_table, normalize_domain_table
from ..logging_utils import get_logger
from ..paths import ensure_dir, get_path, resolve_path
from .structures import download_structure

LOGGER = get_logger(__name__)


def _run_node(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _write_pickle_gz(data, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wb") as handle:
        pickle.dump(data, handle)


def _resolve_existing_script(config, script_value: str) -> str:
    candidate = Path(script_value)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)
    repo_candidate = config.project_root / candidate
    if repo_candidate.exists():
        return str(repo_candidate)
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return str(cwd_candidate)
    raise FileNotFoundError(
        f"ICN3D helper script not found: {script_value}. Install ICN3D separately and set the helper "
        f"script path in config/default.yaml or via CLI."
    )


def _validate_icn3d_prerequisites(config, node_executable: str, sequence_script: str, interaction_script: str, ss_script: str) -> tuple[str, str, str, str]:
    if shutil.which(node_executable) is None:
        raise FileNotFoundError(
            f"Node executable not found on PATH: {node_executable}. Install ICN3D/its JS runtime separately "
            f"and point --node-executable or config.executables.node to the correct command."
        )
    return (
        node_executable,
        _resolve_existing_script(config, sequence_script),
        _resolve_existing_script(config, interaction_script),
        _resolve_existing_script(config, ss_script),
    )


def create_icn3dss(node_executable: str, script_name: str, stru_id: str, icn3d_ss_path: Path) -> None:
    output_path = icn3d_ss_path / f"{stru_id.upper()}_icn3dss.pkl.gz"
    if output_path.exists():
        return
    result = _run_node([node_executable, script_name, stru_id.upper()])
    if result.stdout:
        _write_pickle_gz(json.loads(result.stdout), output_path)


def get_sequence_icn3d(node_executable: str, script_name: str, stru_id: str, sequence_file_path: Path) -> None:
    output_path = sequence_file_path / f"{stru_id.upper()}_sequence.pkl.gz"
    if output_path.exists():
        return
    result = _run_node([node_executable, script_name, stru_id.upper()])
    if result.stdout:
        _write_pickle_gz(json.loads(result.stdout), output_path)


def create_icn3dinteraction(node_executable: str, script_name: str, stru_id: str, chain: str, output_dir: Path) -> None:
    output_path = output_dir / f"{stru_id.upper()}_{chain}_icn3dinteraction.json"
    if output_path.exists():
        return
    subprocess.run([node_executable, script_name, stru_id.upper(), chain, chain, str(output_path)], check=False)


def run_feature_collection(
    config,
    input_table: Path,
    input_dir: str | None = None,
    node_executable: str | None = None,
    sequence_script: str | None = None,
    interaction_script: str | None = None,
    ss_script: str | None = None,
) -> None:
    if not input_table.exists():
        raise FileNotFoundError(f"Input table not found: {input_table}")
    table = normalize_domain_table(load_table(input_table))
    base_input = resolve_path(config, input_dir) if input_dir else get_path(config, "input_dir")
    pdb_dir = ensure_dir(base_input / "pdb_files")
    seq_dir = ensure_dir(base_input / "sequence_file")
    interactions_dir = ensure_dir(base_input / "icn3d_interactions")
    ss_dir = ensure_dir(base_input / "icn3dss")
    node, seq_script, interaction_script, ss_script = _validate_icn3d_prerequisites(
        config,
        node_executable or config.executables.get("node", "node"),
        sequence_script
        or config.executables.get(
            "icn3d_sequence_script",
            "src/gene_ig_identify/integrations/js/get_sequence_icn3d.js",
        ),
        interaction_script
        or config.executables.get(
            "icn3d_interaction_script",
            "src/gene_ig_identify/integrations/js/interactiondetail.js",
        ),
        ss_script
        or config.executables.get(
            "icn3d_secondary_structure_script",
            "src/gene_ig_identify/integrations/js/secondarystructure2.js",
        ),
    )
    seen = {(row["pdb"], row["chainid"]) for _, row in table.iterrows()}
    for pdbid, chain_id in seen:
        LOGGER.info("Collecting ICN3D inputs for %s_%s", pdbid, chain_id)
        download_structure(pdbid, pdb_dir)
        get_sequence_icn3d(node, seq_script, pdbid, seq_dir)
        create_icn3dinteraction(node, interaction_script, pdbid, chain_id, interactions_dir)
        create_icn3dss(node, ss_script, pdbid, ss_dir)
