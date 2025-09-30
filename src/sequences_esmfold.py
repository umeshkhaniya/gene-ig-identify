#!/usr/bin/env python3
import gzip
import pickle
from pathlib import Path
import pandas as pd

def process_sequence_resids(input_excel_file):
    """Process PDB chains, extract residue ranges, and save to gzipped pickle file."""
    all_sequences = {}

    input_data_excel = pd.read_excel(input_excel_file)
    print(input_data_excel.shape)

    for index, row in input_data_excel.iterrows():
        pdb = row["pdb"].upper()  # Replace with actual column name

        chain = row["chainid"]  # Replace with actual column name
        igtype = row["igtype"]  # Replace with actual column name
        begin, end = row["igdomain_res_range"].split("_") 

        # Load sequence data
        seq_file = Path(f"../input/sequence_file/{pdb}_sequence.pkl.gz")
        with gzip.open(seq_file, 'rb') as f:
            seq_info = pickle.load(f)
        
        # Extract residues in range
        res_list = []
        in_range = False
        pdb_chain_info = f"{pdb}_{chain}_{begin}_{end}"
        
        for resid_dict in seq_info[f"{pdb}_{chain}"]:
            resi = str(resid_dict['resi'])
            
            if resi == str(begin):
                in_range = True
            
            if in_range:
                res_list.append((resid_dict['name'], resi))
            
            if resi == str(end):
                break

        if pdb_chain_info in all_sequences:
            print(f"{pdb_chain_info} is already present")
            
        
        all_sequences[pdb_chain_info] = res_list
    
   
    return all_sequences

if __name__ == "__main__":
 
    input_data = "../src_strand/input_data_AF_PDB_igtype_removeORF.xlsx"
    seq_resid_res = process_sequence_resids(input_data)


    print(len(seq_resid_res.keys()))

    # # Save all_sequences to gzipped pickle file
    output_file = "sequences_input_train_test.pkl.gz"
    with gzip.open(output_file, 'wb') as f:
        pickle.dump(seq_resid_res, f)


