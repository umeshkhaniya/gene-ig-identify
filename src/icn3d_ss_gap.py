#!/usr/bin/env python3

import json
import os
import pickle
import gzip


def read_icn3d_ss(ss_file_path, chain_id, begin_res, end_res):
    with gzip.open(ss_file_path, 'rb') as ss_file:
        ss_info = pickle.load(ss_file)

    pdb_chain = f"{os.path.basename(ss_file_path).split('_')[0].upper()}_{chain_id}"

    for ss_chain_res in ss_info['data']:
        if ss_chain_res["chain"] == pdb_chain:
            ss, resn, resi = map(list, (ss_chain_res["secondary"].split(","), ss_chain_res["resn"].split(","), ss_chain_res["resi"].split(",")))
            

            if len(ss) == len(resn) == len(resi):
                try:
                    begin_pos, end_pos = resi.index(begin_res), resi.index(end_res)
                    return ss[begin_pos:end_pos+1], resi[begin_pos:end_pos+1], resn[begin_pos:end_pos+1]
                    
                except ValueError:
                    print(f"residues_id {begin_res} or {end_res} not found in {ss_file_path}.")
            else:
                print(f"Mismatch in lengths of ss, resn, and resi in {ss_file_path}.")
    
    return None, None, None  # Return None if no matching chain is found




def identify_strands_and_gaps(secondary, residues, resnames):
    # Process strands and gaps
    strands = {}
    gaps = {}
    strand_count = 0
    current_strand = []
    previous_end = None

    residue_index = {f"{resnames[i]}{residues[i]}": i for i in range(len(residues))}

    for i, sec in enumerate(secondary):
        res_id = f"{resnames[i]}{residues[i]}"  # Combine resn and resnum

        if sec == "E begin":
            if current_strand:  # Store the previous strand if it exists
                strand_count += 1
                strands[f"strand_{strand_count}"] = current_strand
                previous_end = current_strand[-1]
                current_strand = []
            current_strand.append(res_id)

        elif sec == "E" or sec == "E end":
            current_strand.append(res_id)
            if sec == "E end" or (i == len(secondary) - 1 and current_strand):  # End strand at "E end" or last residue if strand open
                strand_count += 1
                strands[f"strand_{strand_count}"] = current_strand
                if previous_end and previous_end[1:] != current_strand[0][1:]:  # Gap exists
                    gap_key = f"strand_{strand_count-1}_strand_{strand_count}_gap"
                    gap_residues = [f"{resnames[j]}{residues[j]}" for j in range(residue_index[previous_end] + 1, residue_index[current_strand[0]])]
                    gaps[gap_key] = gap_residues if gap_residues else []
                previous_end = current_strand[-1]
                current_strand = []

    # Handle any remaining strand if it wasn't closed. Some 
    if current_strand:
        strand_count += 1
        strands[f"strand_{strand_count}"] = current_strand
        if previous_end and previous_end[1:] != current_strand[0][1:]:  # Gap exists
            gap_key = f"strand_{strand_count-1}_strand_{strand_count}_gap"
            gap_residues = [f"{resnames[j]}{residues[j]}" for j in range(residue_index[previous_end] + 1, residue_index[current_strand[0]])]
            gaps[gap_key] = gap_residues if gap_residues else []

    return strands, gaps

if __name__ == "__main__":
    # begin_res = "433"
    # end_res = "550"
    # ss_file_path = "../input/icn3dss/7YVD_icn3dss.pkl.gz"
    # chain_id = "A"

    # begin_res = "3"
    # end_res = "550"
    # # ss_file_path = "../input/icn3dss/7YVD_icn3dss.pkl.gz"
    # # chain_id = "A"


    begin_res = "211"
    end_res = "295"
    ss_file_path = "../input/icn3dss/1CS6_icn3dss.pkl.gz"
    chain_id = "A"




   
    #ss_file_path = "../input/icn3dss/1CD8_icn3dss.pkl.gz"

    # strand residues_id A3 means A is chain and 3 is resid
    
 
    ss_sel, resi_sel, resn_sel = read_icn3d_ss(ss_file_path, chain_id, begin_res, end_res)
    print(list(zip(ss_sel, resi_sel, resn_sel)))



    
    strands, gaps = identify_strands_and_gaps1(ss_sel, resi_sel, resn_sel)
    print(strands, gaps)

    