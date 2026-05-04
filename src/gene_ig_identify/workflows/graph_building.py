"""Graph creation workflow."""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path

from Bio.PDB import Polypeptide
import h5py
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from ..constants import EXPECTED_EDGE_FEATURES, EXPECTED_NODE_FEATURES
from ..features.hbonds import hbond_icn3d_parser
from ..features.residue_encoding import cys_onehot, onehot_aa_properties, onehot_aa_type, onehot_aromaticity2, optimized_sequence_distance
from ..features.secondary_structure import onehot_ss3, read_icn3d_ss
from ..features.structure_geometry import ResidueAnalyzer
from ..io.artifacts import save_torch
from ..io.tables import load_table, normalize_domain_table
from ..labels import LABEL_MAPPING
from ..logging_utils import get_logger

LOGGER = get_logger(__name__)


def _optional_string(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _graph_name(pdb: str, chain: str, begin_res: str, end_res: str, template_name: str | None, igtype: str | None) -> str:
    parts = [pdb, chain, begin_res, end_res]
    if template_name:
        parts.append(template_name)
    if igtype:
        parts.append(igtype)
    return "_".join(parts)


def create_graph_node_edge(input_excel_file, pdb_file_path, icn3dss_path, contact_file_path, icn3d_interactions, esm2_embedding_file, label_mapping):
    if not Path(input_excel_file).exists():
        raise FileNotFoundError(f"Input table not found: {input_excel_file}")
    if not Path(esm2_embedding_file).exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {esm2_embedding_file}. Generate embeddings separately and pass the file path here."
        )
    input_data_excel = normalize_domain_table(load_table(input_excel_file))
    all_graphs = []
    graph_lookup = {}
    with h5py.File(esm2_embedding_file, "r") as esmh5:
        for _, row in input_data_excel.iterrows():
            pdb = row["pdb"].upper()
            chain = row["chainid"]
            igtype = _optional_string(row.get("ig_type"))
            if igtype is not None and igtype not in label_mapping:
                LOGGER.warning("Skipping unsupported label %s for %s", igtype, row["pdbid_chain"])
                continue
            begin_res, end_res = row["igdomain_res_range"].split("_")
            template_name = _optional_string(row.get("refpdbname"))
            unique_name_file = _graph_name(pdb, chain, begin_res, end_res, template_name, igtype)
            pdb_file = f"{pdb_file_path}{pdb}"
            icn3ss_file = f"{icn3dss_path}{pdb}_icn3dss.pkl.gz"
            icn3d_hbond_file = f"{icn3d_interactions}{pdb}_{chain}_icn3dinteraction.json"
            contact_file = f"{contact_file_path}{pdb}_{chain}_{begin_res}_{end_res}_structure.pkl.gz"
            for required_path, description in (
                (Path(f"{pdb_file}.pdb"), "PDB structure"),
                (Path(icn3ss_file), "ICN3D secondary-structure file"),
                (Path(icn3d_hbond_file), "ICN3D interaction file"),
                (Path(contact_file), "structure feature file"),
            ):
                if description == "PDB structure":
                    cif_path = Path(f"{pdb_file}.cif")
                    if not required_path.exists() and not cif_path.exists():
                        raise FileNotFoundError(
                            f"Missing {description} for {pdb} at {required_path} or {cif_path}. "
                            f"Place the prerequisite files in the configured folders before graph building."
                        )
                elif not required_path.exists():
                    raise FileNotFoundError(
                        f"Missing {description} for {unique_name_file}: {required_path}. "
                        f"Place/generated prerequisite files in the configured folders before graph building."
                    )
            ss_sel, resi_sel, resn_sel = read_icn3d_ss(icn3ss_file, chain, begin_res, end_res)
            if ss_sel is None:
                raise ValueError(
                    f"Secondary structure could not be loaded for {unique_name_file} from {icn3ss_file}."
                )
            resid_to_ss = {f"{resi}_{resn}": ss[0] for resi, resn, ss in zip(resi_sel, resn_sel, ss_sel)}
            with gzip.open(contact_file, "rb") as handle:
                contact_structure_info = pickle.load(handle)
            esmkey_match = f"{pdb}_{chain}_{begin_res}_{end_res}"
            if esmkey_match not in esmh5:
                raise KeyError(
                    f"Protein ID {esmkey_match} not found in embeddings file {esm2_embedding_file}."
                )
            protein_group = esmh5[esmkey_match]
            icn3d_hbond = hbond_icn3d_parser(icn3d_hbond_file)
            res_analyzer = ResidueAnalyzer(pdb_file, chain)
            selected_residues = res_analyzer.get_valid_residues(begin_res, end_res)
            edge_index = []
            edge_attr = []
            node_attr = []
            resid_to_index = {f"{res.id[1]}{res.id[2].strip()}_{Polypeptide.protein_letters_3to1.get(res.get_resname(), 'X')}": idx for idx, res in enumerate(selected_residues)}
            for source_res in selected_residues:
                res_single_letter = Polypeptide.protein_letters_3to1.get(source_res.get_resname(), "X")
                source_node_key = f"{source_res.id[1]}{source_res.id[2].strip()}_{res_single_letter}"
                source_idx = resid_to_index[source_node_key]
                aa_type_onehot = onehot_aa_type(res_single_letter)
                aa_property_onehot = onehot_aa_properties(res_single_letter)
                aromatic_onehot = onehot_aromaticity2(res_single_letter)
                ss = resid_to_ss.get(source_node_key, "c")
                ss_onehot = onehot_ss3(ss)
                source_node_data = contact_structure_info.get(source_node_key)
                if source_node_data is None or source_node_key not in protein_group:
                    raise KeyError(
                        f"Missing node data for {source_node_key} in graph prerequisites for {unique_name_file}."
                    )
                phi_psi_feature = np.array([source_node_data["dihedral_angles"]["phi"], source_node_data["dihedral_angles"]["psi"]])
                tetrahedral_geometry_features = source_node_data["tetrahedral_geometry"]
                forward_calpha_features = np.array(source_node_data["ca_unit_vectors"]["forward"])
                reverse_calpha_features = np.array(source_node_data["ca_unit_vectors"]["reverse"])
                embedding = protein_group[source_node_key][()]
                node_features = np.hstack((aa_type_onehot, aa_property_onehot, aromatic_onehot, ss_onehot, phi_psi_feature, tetrahedral_geometry_features, forward_calpha_features, reverse_calpha_features, embedding))
                if len(node_features) != EXPECTED_NODE_FEATURES:
                    raise ValueError(
                        f"Unexpected node feature length for {unique_name_file}: got {len(node_features)}, "
                        f"expected {EXPECTED_NODE_FEATURES}."
                    )
                node_attr.append(torch.tensor(node_features, dtype=torch.float))
                hbond_source = icn3d_hbond.get(source_node_key, {})
                cb_contacts_info = source_node_data.get("cb_contacts", [])
                for target_node_info in cb_contacts_info:
                    target_resid = target_node_info["residue"]
                    target_idx = resid_to_index.get(target_resid)
                    if target_idx is None:
                        continue
                    ca_unit_vector = target_node_info["ca_unit_vector"]
                    cys_feature = cys_onehot(res_single_letter, target_resid.split("_")[-1])
                    hbond_onehot = [1, 0] if hbond_source is not None and isinstance(hbond_source, set) and target_resid in hbond_source else [0, 1]
                    seq_dist_feature = optimized_sequence_distance(target_node_info["distance"], source_idx, target_idx)
                    edge_features = ca_unit_vector + [seq_dist_feature] + hbond_onehot + cys_feature.tolist()
                    if len(edge_features) != EXPECTED_EDGE_FEATURES:
                        continue
                    edge_index.append([source_idx, target_idx])
                    edge_attr.append(edge_features)
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).T if edge_index else torch.empty((2, 0), dtype=torch.long)
            edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float) if edge_attr else torch.empty((0, EXPECTED_EDGE_FEATURES), dtype=torch.float)
            node_attr_tensor = torch.stack(node_attr) if node_attr else torch.empty((0, EXPECTED_NODE_FEATURES), dtype=torch.float)
            data_kwargs = {
                "x": node_attr_tensor,
                "edge_index": edge_index_tensor,
                "edge_attr": edge_attr_tensor,
            }
            if igtype is not None:
                data_kwargs["y"] = torch.tensor([label_mapping[igtype]], dtype=torch.long)
            data = Data(**data_kwargs)
            data.unique_name_file = unique_name_file
            if template_name is not None:
                data.template = template_name
            data.source_row_index = int(row["_row_order"])
            graph_lookup[unique_name_file] = data
            all_graphs.append(data)
    return all_graphs, graph_lookup


def run(config, input_table: Path, pdb_dir: Path, icn3dss_dir: Path, structure_features_dir: Path, icn3d_interactions_dir: Path, embeddings_file: Path, graphs_output: Path, graph_lookup_output: Path) -> None:
    for required_dir, label in (
        (pdb_dir, "PDB directory"),
        (icn3dss_dir, "ICN3D secondary-structure directory"),
        (structure_features_dir, "structure-features directory"),
        (icn3d_interactions_dir, "ICN3D interactions directory"),
    ):
        if not required_dir.exists():
            raise FileNotFoundError(
                f"{label} not found: {required_dir}. Place the prerequisite files in this folder before running "
                f"`gene-ig-identify graphs build`."
            )
    all_graphs, graph_lookup = create_graph_node_edge(
        input_table,
        f"{pdb_dir}/",
        f"{icn3dss_dir}/",
        f"{structure_features_dir}/",
        f"{icn3d_interactions_dir}/",
        embeddings_file,
        LABEL_MAPPING,
    )
    save_torch(all_graphs, graphs_output)
    save_torch(graph_lookup, graph_lookup_output)
    LOGGER.info("Saved %s graphs to %s and lookup to %s", len(all_graphs), graphs_output, graph_lookup_output)
