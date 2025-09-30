#!/usr/bin/env python3
from Bio.PDB import PDBParser, Polypeptide, Vector, MMCIFParser, calc_dihedral,PDBIO, StructureBuilder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import os
import math


def load_structure(pdb_id):
    """Load structure from either PDB or CIF file"""
    pdb_file = f"{pdb_id}.pdb"
    cif_file = f"{pdb_id}.cif"
    if os.path.exists(cif_file):
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("protein", cif_file)
    elif os.path.exists(pdb_file):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_file)
    else:
        raise FileNotFoundError(f"No structure file found for {pdb_file}")
    return structure

def calculate_dihedral_angles(residues, format_residue_key):
    """Calculate dihedral angles (phi, psi) for a list of residues"""
    dihedral_angles = {}
    for i, residue in enumerate(residues):
        res_key = format_residue_key(residue)
        phi = 0
        psi = 0
        if i > 0 and all(atom in residues[i-1] for atom in ['C']) and \
           all(atom in residue for atom in ['N', 'CA', 'C']):
            v1 = Vector(residues[i-1]['C'].coord)
            v2 = Vector(residue['N'].coord)
            v3 = Vector(residue['CA'].coord)
            v4 = Vector(residue['C'].coord)
            phi_calc = np.degrees(calc_dihedral(v1, v2, v3, v4))
            phi = 0 if np.isnan(phi_calc) else phi_calc
        if i < len(residues)-1 and all(atom in residue for atom in ['N', 'CA', 'C']) and \
           'N' in residues[i+1]:
            v1 = Vector(residue['N'].coord)
            v2 = Vector(residue['CA'].coord)
            v3 = Vector(residue['C'].coord)
            v4 = Vector(residues[i+1]['N'].coord)
            psi_calc = np.degrees(calc_dihedral(v1, v2, v3, v4))
            
            psi = 0 if np.isnan(psi_calc) else psi_calc
        dihedral_angles[res_key] = {'phi': np.cos(np.radians(phi)), 'psi': np.cos(np.radians(psi))}
    return dihedral_angles
def calculate_tetrahedral_geometry(residues, format_residue_key):

    """Calculate tetrahedral geometry (Cβ direction vectors) for a list of residues
    Code borrow and modfied from: 
    https://github.com/drorlab/gvp/blob/master/src/models.py
    https://github.com/Bhattacharya-Lab/EquiPPIS/blob/main/Preprocessing/imputed_tetrahedral_geom_feature.py

    """
    thg = [[[0,0,0],[0,0,0],[0,0,0]] for _ in range(len(residues))]
    thg_val = [[0, 0, 0] for _ in range(len(residues))]
    
    # Populate coordinates
    for i, res in enumerate(residues):
        if 'CA' in res: thg[i][0] = res['CA'].coord
        if 'C' in res: thg[i][1] = res['C'].coord
        if 'N' in res: thg[i][2] = res['N'].coord
    
    # Calculate tetrahedral geometry
    for i in range(len(thg_val)):
        N = np.array(thg[i][2])  # N atom
        Ca = np.array(thg[i][0])  # Cα atom
        C = np.array(thg[i][1])   # C atom
        n = N - Ca
        c = C - Ca
        cross = np.cross(n, c)
        norm_sq = cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2
        
        if norm_sq == 0:
            # If cross product is zero (parallel vectors or invalid coords), skip calculation
            t1 = np.array([0, 0, 0])
            t2 = np.array([0, 0, 0])
        else:
            t1 = cross / (norm_sq * math.sqrt(3))
            summ = n + c
            summ_norm_sq = summ[0] ** 2 + summ[1] ** 2 + summ[2] ** 2
            if summ_norm_sq == 0:
                t2 = np.array([0, 0, 0])
            else:
                t2 = math.sqrt(2/3) * summ / summ_norm_sq
        
        thg_val[i] = t1 - t2
        if np.isnan(thg_val[i]).any():
            thg_val[i] = [0, 0, 0]
    
    return {format_residue_key(res): thg_val[i] for i, res in enumerate(residues)}


def get_cbeta_coordinates(residues, format_residue_key):
    """Retrieve Cβ coordinates, falling back to Cα if Cβ not found. If Ca not then N"""
    cb_coords = {}
    for res in residues:
        res_key = format_residue_key(res)
        if 'CB' in res:
            cb_coords[res_key] = res['CB'].coord.tolist()
        elif 'CA' in res:
            cb_coords[res_key] = res['CA'].coord.tolist()
        elif 'N' in res:
            cb_coords[res_key] = res['N'].coord.tolist()

        else:
            cb_coords[res_key] = [0, 0, 0]
    return cb_coords

def get_calpha_coordinates(residues, format_residue_key):
    """Retrieve Cα coordinates, falling back to Cβ  if Cα not found. If Cα not then N"""
    ca_coords = {}
    for res in residues:
        res_key = format_residue_key(res)
        if 'CA' in res:
            ca_coords[res_key] = res['CA'].coord.tolist()
        elif 'CB' in res:
            ca_coords[res_key] = res['CB'].coord.tolist()
        elif 'N' in res:
            ca_coords[res_key] = res['N'].coord.tolist()

        else:
            ca_coords[res_key] = [0, 0, 0]
    return ca_coords

def calculate_ca_unit_vectors(residues, format_residue_key):
    """Calculate forward and reverse Cα unit vectors for a list of residues

     Code borrow and modfied from: 
     https://github.com/drorlab/gvp/blob/master/src/models.py
     https://github.com/Bhattacharya-Lab/EquiPPIS/blob/main/Preprocessing/prepare_pdb_feature_rev_frwd_ca.py
    """
    unit_vectors = {}
    for i in range(len(residues)):
        res_key = format_residue_key(residues[i])
        u_forw = [0, 0, 0]
        u_revw = [0, 0, 0]
        if 'CA' not in residues[i]:
            unit_vectors[res_key] = {'forward': u_forw, 'reverse': u_revw}
            continue
        cai = np.array(residues[i]['CA'].coord)
        if i < len(residues) - 1 and 'CA' in residues[i + 1]:
            cai_post = np.array(residues[i + 1]['CA'].coord)
            forw = np.subtract(cai_post, cai)
            forwn = np.linalg.norm(forw)
            if forwn > 0:
                u_forw = (forw / forwn).tolist()
        if i > 0 and 'CA' in residues[i - 1]:
            cai_prev = np.array(residues[i - 1]['CA'].coord)
            revw = np.subtract(cai_prev, cai)
            revwn = np.linalg.norm(revw)
            if revwn > 0:
                u_revw = (revw / revwn).tolist()
        unit_vectors[res_key] = {'forward': u_forw, 'reverse': u_revw}
    return unit_vectors

class ResidueAnalyzer:
    def __init__(self, pdb_id, chain_id):
        self.pdb_id = pdb_id
        self.chain_id = chain_id
        self.structure = load_structure(self.pdb_id)
        self.chain = self.structure[0][self.chain_id]
        self.polypeptide = Polypeptide.Polypeptide(self.chain)
        residues = []
        
        
        residues = [res for res in self.chain if Polypeptide.is_aa(res, standard=True)]
        
        res_ids = [f"{r.id[1]}{r.id[2].strip()}" for r in residues]

    def _parse_residue_id(self, res_id):
        """Parse residue ID which could be int or string with insertion code"""
        if res_id.isdigit():
            return int(res_id), ''
        try:
            num = int(res_id[:-1])
            ins_code = res_id[-1]
            return num, ins_code
        except (ValueError, IndexError):
            return int(res_id), ''  # Fallback to integer if parsing fails

    def _format_residue_key(self, residue):
        """Format residue key as number[insertion]_aa (e.g., 27B_A or 27_A)"""
        res_num = residue.id[1]
        ins_code = residue.id[2].strip()
        aa = Polypeptide.protein_letters_3to1.get(residue.get_resname(), "")
        return f"{res_num}{ins_code}_{aa}" if ins_code else f"{res_num}_{aa}"

    def _is_in_range(self, res_num, res_ins, from_num, from_ins, to_num, to_ins):
        """Check if a residue is within the specified range"""
        res = (res_num, res_ins if res_ins else ' ')
        from_res = (from_num, from_ins if from_ins else ' ')
        to_res = (to_num, to_ins if to_ins else ' ')
        return from_res <= res <= to_res

    def _compare_residue_ids(self, res1, res2):
        """Compare two residues to determine which has the higher ID"""
        num1, ins1 = res1.id[1], res1.id[2].strip() or ' '
        num2, ins2 = res2.id[1], res2.id[2].strip() or ' '
        return (num1, ins1) > (num2, ins2)

    def get_valid_residues(self, from_residue, to_residue):
        """Return a list of valid residues within the specified range"""
        from_num, from_ins = self._parse_residue_id(from_residue)
        to_num, to_ins = self._parse_residue_id(to_residue)
        
        valid_residues = [
            res for res in self.chain 
            if Polypeptide.is_aa(res, standard=True) and 
            self._is_in_range(res.id[1], res.id[2].strip(), from_num, from_ins, to_num, to_ins)
        ]
        return valid_residues

    def write_pdb_from_residues(self, from_residue, to_residue, output_pdb_path):
        """Write a new PDB file containing residues between from_residue and to_residue"""
        # Get the valid residues
        valid_residues = self.get_valid_residues(from_residue, to_residue)
        
        # Create a new structure
        new_structure = StructureBuilder.Structure('new_structure')
        new_model = StructureBuilder.Model(0)
        new_chain = StructureBuilder.Chain(self.chain_id)
        
        # Add residues to the new chain
        for residue in valid_residues:
            new_chain.add(residue.copy())  # Copy the residue to avoid modifying the original
        
        # Build the hierarchy
        new_model.add(new_chain)
        new_structure.add(new_model)
        
        # Write to PDB file
        io = PDBIO()
        io.set_structure(new_structure)
        io.save(output_pdb_path)
        
        print(f"Wrote PDB file with residues {from_residue} to {to_residue} to {output_pdb_path}")

    def find_sequences_within_range(self, from_residue, to_residue):
        from_num, from_ins = self._parse_residue_id(from_residue)
        to_num, to_ins = self._parse_residue_id(to_residue)
        sequences_within_range = []
        current_sequence = []
        for residue in self.chain:
            res_num, res_ins = residue.id[1], residue.id[2].strip()
            if (self._is_in_range(res_num, res_ins, from_num, from_ins, to_num, to_ins) and 
                Polypeptide.is_aa(residue, standard=True)):
                current_sequence.append(
                    Polypeptide.protein_letters_3to1.get(residue.get_resname(), "")
                )
            elif current_sequence:
                sequences_within_range.append("".join(current_sequence))
                current_sequence = []
        if current_sequence:
            sequences_within_range.append("".join(current_sequence))
        return sequences_within_range

    def find_ca_contacts_within_range(self, from_residue, to_residue, distance_cutoff):
        """Find Cα contacts within distance_cutoff, unit vector from higher to lower residue ID"""
        from_num, from_ins = self._parse_residue_id(from_residue)
        to_num, to_ins = self._parse_residue_id(to_residue)
        
        valid_residues = self.get_valid_residues(from_residue, to_residue)

         # Coordinates for distance calculation ( Cα or Cβ or N)
        
        # Coordinates for unit vectors (Cβ preferred, Cα fallback)
        ca_or_cb_or_n_coords = []
        for res in valid_residues:
            if 'CA' in res:
                ca_or_cb_or_n_coords.append(res['CA'].coord)
            elif 'CB' in res:
                
                ca_or_cb_or_n_coords.append(res['CB'].coord)
            elif 'N' in res:
                ca_or_cb_or_n_coords.append(res['N'].coord)
            else:
                ca_or_cb_or_n_coords.append(np.array([0, 0, 0]))
       
        ca_or_cb_or_n_coords = np.array(ca_or_cb_or_n_coords)

        residue_keys = [self._format_residue_key(res) for res in valid_residues]
        
        diff = ca_or_cb_or_n_coords[:, np.newaxis] - ca_or_cb_or_n_coords[np.newaxis, :]
        distances = np.linalg.norm(diff, axis=2)
        cos_sim = cosine_similarity(ca_or_cb_or_n_coords)
        
        contacts_within_range = {key: [] for key in residue_keys}


        
        for i, (res1_key, res1) in enumerate(zip(residue_keys, valid_residues)):
            contact_indices = np.where(
                (distances[i] <= distance_cutoff) & (distances[i] > 0)
            )[0]
            ca1 = ca_or_cb_or_n_coords[i]
            for j in contact_indices:
                res2 = valid_residues[j]
                res2_key = residue_keys[j]
                res2_num_ins = res2_key.split('_')[0]
                res2_aa = res2_key.split('_')[1]
                ca2 = ca_or_cb_or_n_coords[j]
                # Unit vector: higher residue ID to lower residue ID
                if self._compare_residue_ids(res1, res2):
                    vector = ca1 - ca2  # res1 > res2
                else:
                    vector = ca2 - ca1  # res2 > res1
                norm = np.linalg.norm(vector)
                unit_vector = (vector / norm).tolist() if norm > 0 else [0, 0, 0]
                contacts_within_range[res1_key].append({
                    'residue': f"{res2_num_ins}_{res2_aa}",
                    'cosine_similarity': cos_sim[i][j],
                    'distance': distances[i][j],
                    'ca_unit_vector': unit_vector
                })
                
        return contacts_within_range




    def find_cb_contacts_within_range(self, from_residue, to_residue, distance_cutoff):
        """Find Cβ contacts within distance_cutoff, with Cα unit vector from higher to lower residue ID"""
        from_num, from_ins = self._parse_residue_id(from_residue)
        to_num, to_ins = self._parse_residue_id(to_residue)
        
        valid_residues = self.get_valid_residues(from_residue, to_residue)
        # Coordinates for distance calculation (Cβ preferred, Cα fallback, N as last resort)
        coords = []
        # Coordinates for Cα unit vectors
        ca_coords = []
        for res in valid_residues:
            # For distance calculation
            if 'CB' in res:
                coords.append(res['CB'].coord)
            elif 'CA' in res:
                coords.append(res['CA'].coord)
            elif 'N' in res:
                coords.append(res['N'].coord)
            else:
                coords.append(np.array([0, 0, 0]))

            
            # For Cα unit vector calculation
            if 'CA' in res:
                ca_coords.append(res['CA'].coord)
            elif 'CB' in res:
                ca_coords.append(res['CB'].coord)
            elif 'N' in res:
                ca_coords.append(res['N'].coord)
            else:
                ca_coords.append(np.array([0, 0, 0]))  # Fallback if no Cα
            
        coords = np.array(coords)
        ca_coords = np.array(ca_coords)
        
        residue_keys = [self._format_residue_key(res) for res in valid_residues]
        
        diff = coords[:, np.newaxis] - coords[np.newaxis, :]
        distances = np.linalg.norm(diff, axis=2)
        cos_sim = cosine_similarity(coords)
        
        contacts_within_range = {key: [] for key in residue_keys}
        
        for i, (res1_key, res1) in enumerate(zip(residue_keys, valid_residues)):
            contact_indices = np.where(
                (distances[i] <= distance_cutoff) & (distances[i] > 0)
            )[0]
            ca1 = ca_coords[i]  # Use Cα coordinates for unit vector
            for j in contact_indices:
                res2 = valid_residues[j]
                res2_key = residue_keys[j]
                res2_num_ins = res2_key.split('_')[0]
                res2_aa = res2_key.split('_')[1]
                ca2 = ca_coords[j]  # Use Cα coordinates for unit vector
                # Unit vector: higher residue ID to lower residue ID using Cα
                if self._compare_residue_ids(res1, res2):
                    vector = ca1 - ca2  # res1 > res2
                else:
                    vector = ca2 - ca1  # res2 > res1
                norm = np.linalg.norm(vector)
                unit_vector = (vector / norm).tolist() if norm > 0 else [0, 0, 0]
                contacts_within_range[res1_key].append({
                    'residue': f"{res2_num_ins}_{res2_aa}",
                    'cosine_similarity': cos_sim[i][j],  #Cbeta
                    'distance': distances[i][j],
                    'ca_unit_vector': unit_vector  # Changed key to reflect Cα unit vector
                })
                
        return contacts_within_range

    
    def analyze_residues(self, from_residue, to_residue, distance_cutoff):
        """Analyze residues and return a dictionary with all geometric features for fast lookup"""
        from_num, from_ins = self._parse_residue_id(from_residue)
        to_num, to_ins = self._parse_residue_id(to_residue)
        
        valid_residues = self.get_valid_residues(from_residue, to_residue)
        residue_data = {self._format_residue_key(res): {
            'dihedral_angles': {'phi': 0, 'psi': 0},
            'tetrahedral_geometry': [0, 0, 0],
            'calpha_coords': [0, 0, 0],
            'cbeta_coords': [0, 0, 0],
            'ca_unit_vectors': {'forward': [0, 0, 0], 'reverse': [0, 0, 0]},
            'ca_contacts': [],
            'cb_contacts': []
        } for res in valid_residues}
        
        dihedral_angles = calculate_dihedral_angles(valid_residues, self._format_residue_key)
        tetrahedral_geometry = calculate_tetrahedral_geometry(valid_residues, self._format_residue_key)
        calpha_coords = get_calpha_coordinates(valid_residues, self._format_residue_key)
        cbeta_coords = get_cbeta_coordinates(valid_residues, self._format_residue_key)
        ca_unit_vectors = calculate_ca_unit_vectors(valid_residues, self._format_residue_key)
        ca_contacts = self.find_ca_contacts_within_range(from_residue, to_residue, distance_cutoff)
        cb_contacts = self.find_cb_contacts_within_range(from_residue, to_residue, distance_cutoff)
        
        for res_key in residue_data:
            residue_data[res_key]['dihedral_angles'] = dihedral_angles.get(res_key, {'phi': 0, 'psi': 0})
            residue_data[res_key]['tetrahedral_geometry'] = tetrahedral_geometry.get(res_key, [0, 0, 0])
            residue_data[res_key]['calpha_coords'] = calpha_coords.get(res_key, [0, 0, 0])
            residue_data[res_key]['cbeta_coords'] = cbeta_coords.get(res_key, [0, 0, 0])
            residue_data[res_key]['ca_unit_vectors'] = ca_unit_vectors.get(res_key, {'forward': [0, 0, 0], 'reverse': [0, 0, 0]})
            residue_data[res_key]['ca_contacts'] = ca_contacts.get(res_key, [])
            residue_data[res_key]['cb_contacts'] = cb_contacts.get(res_key, [])
        
        return residue_data

if __name__ == "__main__":
    # pdb_id = "./5esv"
    # chain_id = "A"
    # from_residue = "2"  
    # to_residue = "100H" 
    # cutoff_distance = 6.0
    pdb_id = "1O6S"
    chain_id = "B"
    from_residue = "7"  
    to_residue = "100"  
    cutoff_distance = 8.0

    analyzer = ResidueAnalyzer(f"../input/pdb_files/{pdb_id}", chain_id)
    sequences = analyzer.find_sequences_within_range(from_residue, to_residue)
    residue_data = analyzer.analyze_residues(from_residue, to_residue, cutoff_distance)
    
    print(f"Sequences: {sequences}")
    print("\nResidue Data:")
    for res_key, data in residue_data.items():
        print(f"{res_key}:")
        print(f"  Dihedral Angles: phi={data['dihedral_angles']['phi']:.2f}, psi={data['dihedral_angles']['psi']:.2f}")
        print(f"  Tetrahedral Geometry: {data['tetrahedral_geometry']}")
        print(f"  Cβ Coordinates: {data['cbeta_coords']}")
        print(f"  Cα Coordinates: {data['calpha_coords']}")
        print(f"  Cα Unit Vectors: forward={data['ca_unit_vectors']['forward']}, reverse={data['ca_unit_vectors']['reverse']}")
        print(f"  Cα Contacts (within {cutoff_distance}Å):")
        for contact in data['ca_contacts']:
            print(f"    {contact['residue']}: cos_sim={contact['cosine_similarity']:.3f}, "
                  f"dist={contact['distance']:.3f}, ca_unit_vector={contact['ca_unit_vector']}")
        print(f"  Cβ Contacts (within {cutoff_distance}Å, fallback to Cα if Cβ missing, Cα unit vector):")
        for contact in data['cb_contacts']:
            print(f"    {contact['residue']}: cos_sim={contact['cosine_similarity']:.3f}, "
                  f"dist={contact['distance']:.3f}, ca_unit_vector={contact['ca_unit_vector']}")
    
    # Plotting remains unchanged
    plt.figure(figsize=(12, 8))
    residue_numbers = [float(key.split('_')[0][:-1]) if key.split('_')[0][-1].isalpha() 
                      else float(key.split('_')[0]) for key in residue_data.keys()]
    ca_contact_counts = [len(data['ca_contacts']) for data in residue_data.values()]
    cb_contact_counts = [len(data['cb_contacts']) for data in residue_data.values()]

    plt.scatter(residue_numbers, ca_contact_counts, c='blue', label='Cα Contacts', alpha=0.5)
    plt.scatter(residue_numbers, cb_contact_counts, c='red', label='Cβ Contacts', alpha=0.5)
    plt.legend()
    plt.xlabel('Residues')
    plt.ylabel('Number of Contacts')
    plt.title('Residue Contact Analysis (within 6Å)')
    plt.tight_layout()
    #plt.show()

