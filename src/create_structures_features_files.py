#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import pickle
import gzip
from typing import Union

import structure_features as structure_features


def process_pdb_files(
    input_excel: Union[str, Path],
    input_dir: Union[str, Path],
    pdb_subdir: str,
    structure_subdir: str,
    cutoff_distance: int
) -> None:
    """
    Process PDB files and generate contact and phi/psi angle files.
    
    Args:
        input_excel: Path to the input Excel file
        input_dir: Base directory containing input/output subdirectories
        pdb_subdir: Subdirectory for PDB files
        structure_subdir: Subdirectory for residues features
    """
    try:
        # Convert to Path objects
        input_path = Path(input_dir)
        excel_path = Path(input_excel)
        
        # Read input data
        if not excel_path.is_file():
            raise FileNotFoundError(f"Excel file not found: {excel_path}")
        input_data = pd.read_excel(excel_path)
        
        # Create column set
        column_set = input_data[['pdbid_chain', 'igdomain_res_range']].itertuples(index=False)
        
        # Setup directory paths
        pdb_path = input_path / pdb_subdir
        structure_features_path = input_path / structure_subdir
        
        
        # Ensure directories exist
        for directory in (pdb_path, structure_features_path):
            directory.mkdir(parents=True, exist_ok=True)

        for pdb_chain, res_range in column_set:
            print(f"Processing {pdb_chain}:{res_range}")
            
            try:
                domain_begin, domain_end = res_range.split("_")
                pdb, chain = pdb_chain.split("_")
            except ValueError as e:
                print(f"Error parsing {pdb_chain} or {res_range}: {e}")
                continue

            if len(chain) != 1:
                continue

            # File paths
            pdb_upper = pdb.upper()
            pdb_file = pdb_path / pdb_upper

            print(pdb_file)


            
            structure_features_file = f"{pdb_upper}_{chain}_{domain_begin}_{domain_end}_structure.pkl.gz"
            
            structure_features_file_path = structure_features_path / structure_features_file 
            

            if structure_features_file_path.exists():
                continue

           
            # for ext in [".pdb", ".cif"]:
            #     print(pdb_file.with_suffix(ext))

            pdb_exists = any((pdb_file.with_suffix(ext)).exists() for ext in [".pdb", ".cif"])
            if not pdb_exists:
                print(f"PDB/CIF file not found for {pdb_upper}")
                continue

            try:
                analyzer = structure_features.ResidueAnalyzer(pdb_file, chain)
                #analyzer = structure_features_cbeta.ResidueAnalyzer(pdb_file, chain, cutoff_distance)

                #residues_features = analyzer.analyze_residues(domain_begin, domain_end, cutoff_distance)

                residues_features = analyzer.analyze_residues(domain_begin, domain_end, cutoff_distance)
    
                
                if residues_features:
                    with gzip.open(structure_features_file_path , "wb") as f:
                        pickle.dump(residues_features, f)
                else:
                    print(f"No residues structure features are not found for {contact_file} - chain {chain} may not exist")

            except Exception as e:
                print(f"Error processing {pdb_chain}: {e}")

    except FileNotFoundError as e:
        print(f"Input file error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

# Example usage:
if __name__ == "__main__":
    # Now all parameters are required
    # process_pdb_files(
    #     input_excel="../src_strand/input_dataset_remove_CD19_ORF_IgFn3like.xlsx",
    #     input_dir="../input",
    #     pdb_subdir="pdb_files",
    #     structure_subdir="structure_features_residues_Cbeta",
    #     cutoff_distance = 6 #cutoff_distance
    # )

     process_pdb_files(
        input_excel="human_proteome_all_igstrand.xlsx",
        input_dir="../input",
        pdb_subdir="pdb_files",
        structure_subdir="structure_features_residues_proteome",
        cutoff_distance = 8 #cutoff_distance
    )
    
   