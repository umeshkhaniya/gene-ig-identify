#!/usr/bin/env python3
import gzip
import pickle
import os
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from Bio.PDB import Polypeptide
import torch
from torch_geometric.data import Data
import h5py

import icn3d_ss_gap
import residues_features
import ss_features
from structure_features import ResidueAnalyzer
from hbond_icn3d import hbond_icn3d_parser

expected_node_features = 40 #  no coordinates # esm 1280 + 40 
expected_edge_features = 8


def create_graph_node_edge(input_excel_file, pdb_file_path, icn3dss_path, contact_file_path, icn3d_interactions, esm2_embedding_file, label_mapping):

    # input_excel_file is an excel file i.e. has pdb_chain, Igtype etc data.


    input_data_excel = pd.read_excel(input_excel_file)

    print(f"Total Data:{input_data_excel.shape}")

    all_graphs = []
    graph_lookup = {}  # Lookup for later comparison

    # read esmbedding first.

    with h5py.File(esm2_embedding_file, "r") as esmh5:

        for index, row in input_data_excel.iterrows():
            
            pdb = row["pdb"].upper()  # Replace with actual column name

            chain = row["chainid"]  # Replace with actual column name
            #print(pdb, chain) 
            # pdb = row["pdb"].upper()  # Replace with actual column name

            # chain = row["chainid"]  # Replace with actual column name
            igtype = row["igtype"]  # Replace with actual column name
            begin_res, end_res = row["igdomain_res_range"].split("_") 
            template_name = row['refpdbname']

            unique_name_file = f"{pdb}_{chain}_{begin_res}_{end_res}_{row['refpdbname']}_{igtype}"
            print(f"Processing {unique_name_file}")

            pdb_file = f"{pdb_file_path}{pdb}"
            icn3ss_file = f"{icn3dss_path}{pdb}_icn3dss.pkl.gz"
            icn3d_hbond_file = f"{icn3d_interactions}{pdb}_{chain}_icn3dinteraction.json"
            contact_file = f"{contact_file_path}{pdb}_{chain}_{begin_res}_{end_res}_structure.pkl.gz"

            # # open icndss file:
            ss_sel, resi_sel, resn_sel = icn3d_ss_gap.read_icn3d_ss(icn3ss_file, chain, begin_res, end_res)
            # # open contact file

            resid_to_ss = {f"{resi}_{resn}": ss[0] for resi, resn, ss in zip(resi_sel, resn_sel, ss_sel)}


            with gzip.open(contact_file, 'rb') as f:
                contact_structure_info = pickle.load(f)


            # open ESMFOLD files:

            esmkey_match = f"{pdb.upper()}_{chain}_{begin_res}_{end_res}"



            #esm_features = 

            if esmkey_match  not in esmh5:
                print(f"[ERROR] Protein ID '{esmkey_match}' and row: {row} not found in file. Skipping.")
                continue  # Skip this one



            #protein_group = esmh5[esmkey_match]



            # hbond
            icn3d_hbond = hbond_icn3d_parser(icn3d_hbond_file)

            #print(icn3d_hbond)

            res_analyzer = ResidueAnalyzer(pdb_file, chain)

            selected_residues = res_analyzer.get_valid_residues(begin_res, end_res)

            edge_index = []
            edge_attr = []
            node_attr = []

            #single_letter = Polypeptide.protein_letters_3to1.get(res.get_resname(), "X")

            resid_to_index = {f"{res.id[1]}{res.id[2].strip()}_{Polypeptide.protein_letters_3to1.get(res.get_resname(), 'X')}": idx for idx, res in enumerate(selected_residues)}


            for source_res in selected_residues:
                #print(source_res)
                res_single_letter = Polypeptide.protein_letters_3to1.get(source_res.get_resname(), "X")

                source_node_key = f"{source_res.id[1]}{source_res.id[2].strip()}_{res_single_letter}"
                source_idx = resid_to_index[source_node_key]
                
                aa_type_onehot = residues_features.onehot_aa_type(res_single_letter)
                aa_property_onehot = residues_features.onehot_aa_properties(res_single_letter)
                aromatic_onehot = residues_features.onehot_aromaticity2(res_single_letter)


                ss = resid_to_ss.get(source_node_key, "c")
                

                ss_onehot = ss_features.onehot_ss3(ss) # b/c E sometimes has E end like this

               

                # structure data;

              

                source_node_data = contact_structure_info.get(source_node_key)

                

                phi_psi_feature = np.array([source_node_data['dihedral_angles']['phi'], 
                            source_node_data['dihedral_angles']['psi']])

                tetrahedral_geometry_features = source_node_data['tetrahedral_geometry']

                forward_calpha_features = np.array(source_node_data['ca_unit_vectors']['forward'])

                reverse_calpha_features = np.array(source_node_data['ca_unit_vectors']['reverse'])

                #calpha_coords = np.array(source_node_data['calpha_coords'])



                # if source_node_key not in protein_group:
                #     print(f"[ERROR] Protein ID '{esmkey_match}' and row: {row} not found in file. Skipping.")
                #     continue

                # Load embedding
                #embedding = protein_group[source_node_key][()]  # Shape: (1280,)

                # print(embedding)
                # print(len(embedding))

                
                node_features = np.hstack((aa_type_onehot, aa_property_onehot, aromatic_onehot, ss_onehot,phi_psi_feature, 
                                             tetrahedral_geometry_features, forward_calpha_features,reverse_calpha_features))



                # # #print(node_features)
                #print(len(node_features))

                

                if len(node_features) != expected_node_features:
                    print(f"Following  pdb {pdb} and chain {chain} has issue with node features")
                    continue

                node_features = torch.tensor(node_features, dtype=torch.float)
                #print(node_features)
                node_attr.append(node_features)


                hbond_source = icn3d_hbond.get(source_node_key, {})

                
                
                cb_contacts_info = source_node_data.get('cb_contacts', [])
                if cb_contacts_info:
                    for target_node_info in cb_contacts_info:
                        target_resid = target_node_info["residue"]
                        
                        target_idx = resid_to_index.get(target_resid) #sometime alternate conformer can have error so

                        if target_idx is not None:
                        
                            ca_unit_vector = target_node_info["ca_unit_vector"]

                            cys_onehot = residues_features.cys_onehot(res_single_letter, target_resid.split("_")[-1])

                            hbond_onehot = [0, 1] # this 

                            if hbond_source is not None and isinstance(hbond_source, (set)):
                                if target_resid in hbond_source:
                                    hbond_onehot = [1,0]
                            
                            seq_dist_feature = residues_features.optimized_sequence_distance(target_node_info["distance"], source_idx, target_idx)

                            edge_features = ca_unit_vector + [seq_dist_feature] + hbond_onehot + cys_onehot.tolist()

                            if len(edge_features) != expected_edge_features:
                                print(f"Warning: Expected {expected_edge_features} edge features, got {len(edge_features)}")
                                continue

                            edge_index.append([source_idx, target_idx])

                            if len(edge_features) != expected_edge_features:
                                print(f"Following  pdb {pdb} and chain {chain} has issue with edge features")



                            edge_attr.append(edge_features)



            # Convert to tensors
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).T if edge_index else torch.empty((2, 0), dtype=torch.long)
            edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float) if edge_attr else torch.empty((0, expected_edge_features), dtype=torch.float)
            node_attr_tensor = torch.stack(node_attr) if node_attr else torch.empty((0, expected_node_features), dtype=torch.float)
            y_label = torch.tensor([label_mapping[igtype]], dtype=torch.long)

            data = Data(x=node_attr_tensor, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor, y=y_label)

            data.unique_name_file = unique_name_file
            data.template = template_name 
            graph_lookup[unique_name_file] = data  
            all_graphs.append(data)
        print(f"Total graphs created: {len(all_graphs)}")


    return all_graphs, graph_lookup
          
           

if __name__ == "__main__":
    # File paths (can be constants)
    pdb_file_path = "../input/pdb_files/"
    icn3dss_file_path = "../input/icn3dss/"
    contact_file_path = "../input/structure_features_residues/"
    icn3d_interactions = "../input/icn3d_interactions/"


    esm2_embedding_file = "/data/khaniyau2/deep_learning/esm2/esm2_t33_650M_UR50D_embedding/esm2_t33_650M_UR50D_all_embeddings_input_train_test.h5"

    
    LABEL_MAPPING = {'IgV':0, 'IgC1': 1,  'IgC2':2,'IgI':3, 'IgE':4,'Cadherin': 5, 'IgFN3':6,
    'Lamin':7,  'SOD':8,'IgFN3-like':9, 'CD19':10, "JellyRoll":11,'ORF':12} 



    all_graphs, graph_lookup = create_graph_node_edge("../input_data_AF_PDB_igtype_removeORF.xlsx", pdb_file_path, icn3dss_file_path, 
         contact_file_path,icn3d_interactions, esm2_embedding_file, LABEL_MAPPING)
    
    # all_graphs, graph_lookup = create_graph_node_edge("../sample_data.xlsx", pdb_file_path, icn3dss_file_path, 
    #      contact_file_path,icn3d_interactions, esm2_embedding_file, LABEL_MAPPING)



    for i, graph in enumerate(all_graphs):
        node_shape = graph.x.shape
        edge_attr_shape = graph.edge_attr.shape if hasattr(graph, 'edge_attr') else None
        num_nodes = node_shape[0]
        num_edges = edge_attr_shape[0] if edge_attr_shape is not None else 0
        print(f"Graph {i}: x shape {node_shape}")
        print(f"Graph {i}: edge_attr shape {edge_attr_shape}")



    
    torch.save(all_graphs, "all_graphs_nocor_noesm2.pt")
    torch.save(graph_lookup, "graph_lookup_nocor_noesm2.pt")
    
  

  
    

