import json
from typing import Dict, Set
from Bio.PDB import Polypeptide

def hbond_icn3d_parser(file_path: str) -> Dict[str, Set[str]]:
    """
    Process hydrogen bond data from a JSON file and return a bidirectional mapping.
    
    Args:
        file_path: Path to the JSON file containing interaction data
        
    Returns:
        Dictionary mapping residue IDs to sets of hydrogen-bonded residues
    """
    # Initialize result dictionary
    hbond_dict: Dict[str, Set[str]] = {}
    
    # Read JSON data
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    for entry in data['bondCnt']:
        if entry['cntHbond'] <= 0:
            continue
            
        res1_parts = entry['res1'].split("_")
        resid1_aa = f"{res1_parts[2]}_{Polypeptide.protein_letters_3to1.get(res1_parts[3], 'X')}"
        hbond_dict.setdefault(resid1_aa, set())
        
        # Process residue 2 interactions
        for interaction in entry['res2'].split():
            if 'hbond' not in interaction or 'main,main' not in interaction:
                continue
                
            # Extract and process target residue
            target_res = interaction.split(':')[0]
            target_parts = target_res.split("_")
            target_resid_aa = f"{target_parts[2]}_{Polypeptide.protein_letters_3to1.get(target_parts[3], 'X')}"
            
            # Add bidirectional relationships
            hbond_dict[resid1_aa].add(target_resid_aa)
            hbond_dict.setdefault(target_resid_aa, set()).add(resid1_aa)
    
    return hbond_dict

    

if __name__ == "__main__":
    input_file = "../input/icn3d_interactions/2DM3_A_icn3dinteraction.json"
  # Can be any filename
    result = hbond_icn3d_parser(input_file)
    print(result)

# # Optional: Save to JSON file
# with open('hbond_main_main_bidirectional.json', 'w') as f:
#     json.dump({k: list(v) for k, v in hbond_dict.items()}, f, indent=4)