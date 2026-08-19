"""Tests for experiment-specific label mappings in graph building."""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from gene_ig_identify.config import load_config
from gene_ig_identify.labels import LABEL_MAPPING, label_mapping_from_config
from gene_ig_identify.workflows import graph_building


class FakeDataset:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        if key != ():
            raise KeyError(key)
        return self.values


class FakeH5:
    def __init__(self):
        self.groups = {
            "1ABC_A_1_1": {
                "1_A": FakeDataset(np.zeros(1280, dtype=np.float32)),
            }
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def __contains__(self, key):
        return key in self.groups

    def __getitem__(self, key):
        return self.groups[key]


class FakeResidue:
    id = (" ", 1, " ")

    def get_resname(self):
        return "ALA"


class FakeResidueAnalyzer:
    def __init__(self, pdb_file, chain):
        self.pdb_file = pdb_file
        self.chain = chain

    def get_valid_residues(self, begin_res, end_res):
        return [FakeResidue()]


def create_minimal_graph_inputs(base_dir: Path, label_names: list[str]) -> dict[str, Path]:
    pdb_dir = base_dir / "pdb_files"
    ss_dir = base_dir / "icn3dss"
    structure_dir = base_dir / "structure_features_residues"
    interactions_dir = base_dir / "icn3d_interactions"
    output_dir = base_dir / "output"
    for directory in (pdb_dir, ss_dir, structure_dir, interactions_dir, output_dir):
        directory.mkdir(parents=True)

    (pdb_dir / "1ABC.pdb").write_text("HEADER test\n", encoding="utf-8")
    (ss_dir / "1ABC_icn3dss.pkl.gz").write_bytes(b"placeholder")
    (interactions_dir / "1ABC_A_icn3dinteraction.json").write_text("{}", encoding="utf-8")

    residue_features = {
        "1_A": {
            "dihedral_angles": {"phi": 0.0, "psi": 0.0},
            "tetrahedral_geometry": [0.0, 0.0, 0.0],
            "ca_unit_vectors": {
                "forward": [0.0, 0.0, 0.0],
                "reverse": [0.0, 0.0, 0.0],
            },
            "cb_contacts": [],
        }
    }
    with gzip.open(structure_dir / "1ABC_A_1_1_structure.pkl.gz", "wb") as handle:
        pickle.dump(residue_features, handle)

    table_path = base_dir / "domains.csv"
    pd.DataFrame(
        {
            "pdbid_chain": ["1ABC_A"] * len(label_names),
            "igdomain_res_range": ["1_1"] * len(label_names),
            "ig_type": label_names,
        }
    ).to_csv(table_path, index=False)

    embeddings_file = base_dir / "embeddings.h5"
    embeddings_file.write_bytes(b"placeholder")

    return {
        "table": table_path,
        "pdb_dir": pdb_dir,
        "ss_dir": ss_dir,
        "structure_dir": structure_dir,
        "interactions_dir": interactions_dir,
        "embeddings": embeddings_file,
        "graphs_output": output_dir / "graphs.pt",
        "lookup_output": output_dir / "lookup.pt",
    }


def create_graphs_with_mapping(paths: dict[str, Path], label_mapping: dict[str, int]):
    with (
        patch.object(graph_building.h5py, "File", return_value=FakeH5()),
        patch.object(graph_building, "read_icn3d_ss", return_value=(["E"], ["1"], ["A"])),
        patch.object(graph_building, "hbond_icn3d_parser", return_value={}),
        patch.object(graph_building, "ResidueAnalyzer", FakeResidueAnalyzer),
    ):
        return graph_building.create_graph_node_edge(
            paths["table"],
            f"{paths['pdb_dir']}/",
            f"{paths['ss_dir']}/",
            f"{paths['structure_dir']}/",
            f"{paths['interactions_dir']}/",
            paths["embeddings"],
            label_mapping,
        )


class GraphBuildingLabelMappingTests(unittest.TestCase):
    def setUp(self):
        self.project_root = Path(__file__).resolve().parents[1]

    def load_experiment_mapping(self, filename: str) -> dict[str, int]:
        config = load_config(self.project_root / "config" / "experiments" / filename)
        return label_mapping_from_config(config)

    def test_exp00_graph_label_behavior_remains_unchanged(self):
        with TemporaryDirectory() as tmp_dir:
            paths = create_minimal_graph_inputs(Path(tmp_dir), ["IgV", "CD19"])
            graphs, graph_lookup = create_graphs_with_mapping(paths, LABEL_MAPPING)

        self.assertEqual([int(graph.y.item()) for graph in graphs], [0, 7])
        self.assertEqual(set(graph_lookup), {"1ABC_A_1_1_IgV", "1ABC_A_1_1_CD19"})

    def test_exp01_graph_generation_encodes_new_labels(self):
        new_labels = ["IgE", "IgFN3-like", "SOD"]
        exp01_mapping = self.load_experiment_mapping("exp01_12class.yaml")

        with TemporaryDirectory() as tmp_dir:
            paths = create_minimal_graph_inputs(Path(tmp_dir), new_labels)
            graphs, graph_lookup = create_graphs_with_mapping(paths, exp01_mapping)

        self.assertNotIn("ORF", exp01_mapping)
        self.assertEqual([int(graph.y.item()) for graph in graphs], [8, 9, 10])
        self.assertEqual(
            set(graph_lookup),
            {
                "1ABC_A_1_1_IgE",
                "1ABC_A_1_1_IgFN3-like",
                "1ABC_A_1_1_SOD",
            },
        )

    def test_exp02_graph_generation_skips_cd19_label(self):
        exp02_mapping = self.load_experiment_mapping("exp02_7class.yaml")

        with TemporaryDirectory() as tmp_dir:
            paths = create_minimal_graph_inputs(Path(tmp_dir), ["CD19"])
            with self.assertLogs(graph_building.LOGGER, level="WARNING") as captured:
                with (
                    patch.object(graph_building.h5py, "File", return_value=FakeH5()),
                    patch.object(graph_building, "read_icn3d_ss") as read_ss,
                ):
                    graphs, graph_lookup = graph_building.create_graph_node_edge(
                        paths["table"],
                        f"{paths['pdb_dir']}/",
                        f"{paths['ss_dir']}/",
                        f"{paths['structure_dir']}/",
                        f"{paths['interactions_dir']}/",
                        paths["embeddings"],
                        exp02_mapping,
                    )

        self.assertEqual(graphs, [])
        self.assertEqual(graph_lookup, {})
        read_ss.assert_not_called()
        self.assertIn("Skipping unsupported label CD19", "\n".join(captured.output))

    def test_run_uses_default_exp00_mapping_from_config(self):
        config = load_config(self.project_root / "config" / "default.yaml")

        with TemporaryDirectory() as tmp_dir:
            paths = create_minimal_graph_inputs(Path(tmp_dir), ["IgV"])
            with (
                patch.object(graph_building, "create_graph_node_edge", return_value=(["graph"], {"graph": "graph"})) as create_graph,
                patch.object(graph_building, "save_torch") as save_torch,
            ):
                graph_building.run(
                    config,
                    input_table=paths["table"],
                    pdb_dir=paths["pdb_dir"],
                    icn3dss_dir=paths["ss_dir"],
                    structure_features_dir=paths["structure_dir"],
                    icn3d_interactions_dir=paths["interactions_dir"],
                    embeddings_file=paths["embeddings"],
                    graphs_output=paths["graphs_output"],
                    graph_lookup_output=paths["lookup_output"],
                )

        self.assertEqual(create_graph.call_args.args[-1], LABEL_MAPPING)
        self.assertEqual(save_torch.call_count, 2)

    def test_run_accepts_explicit_label_mapping(self):
        config = load_config(self.project_root / "config" / "default.yaml")
        custom_mapping = {"CustomLabel": 0}

        with TemporaryDirectory() as tmp_dir:
            paths = create_minimal_graph_inputs(Path(tmp_dir), ["CustomLabel"])
            with (
                patch.object(graph_building, "create_graph_node_edge", return_value=([], {})) as create_graph,
                patch.object(graph_building, "save_torch"),
            ):
                graph_building.run(
                    config,
                    input_table=paths["table"],
                    pdb_dir=paths["pdb_dir"],
                    icn3dss_dir=paths["ss_dir"],
                    structure_features_dir=paths["structure_dir"],
                    icn3d_interactions_dir=paths["interactions_dir"],
                    embeddings_file=paths["embeddings"],
                    graphs_output=paths["graphs_output"],
                    graph_lookup_output=paths["lookup_output"],
                    label_mapping=custom_mapping,
                )

        self.assertEqual(create_graph.call_args.args[-1], custom_mapping)


if __name__ == "__main__":
    unittest.main()
