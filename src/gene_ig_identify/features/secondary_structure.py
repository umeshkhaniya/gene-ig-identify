#!/usr/bin/env python3

import gzip
import os
import pickle
import numpy as np

ss3_map = {
    "H": "H", "B": "C", "E": "E", "G": "H",
    "I": "C", "T": "C", "S": "C", "-": "C",
}


def onehot_ss3(ss):
    ss3_onehot = np.zeros(3, dtype=int)
    if ss in ("H", "G"):
        ss3_onehot[0] = 1
    elif ss in ("E"):
        ss3_onehot[1] = 1
    else:
        ss3_onehot[2] = 1
    return ss3_onehot


def onehot_sa3(sa):
    sa_one_hot = np.zeros(3, dtype=int)
    if sa <= 0.3:
        sa_one_hot[0] = 1
    elif sa <= 0.6:
        sa_one_hot[1] = 1
    else:
        sa_one_hot[2] = 1
    return sa_one_hot


def read_icn3d_ss(ss_file_path, chain_id, begin_res, end_res):
    with gzip.open(ss_file_path, "rb") as ss_file:
        ss_info = pickle.load(ss_file)
    pdb_chain = f"{os.path.basename(ss_file_path).split('_')[0].upper()}_{chain_id}"
    for ss_chain_res in ss_info["data"]:
        if ss_chain_res["chain"] == pdb_chain:
            ss, resn, resi = map(list, (ss_chain_res["secondary"].split(","), ss_chain_res["resn"].split(","), ss_chain_res["resi"].split(",")))
            if len(ss) == len(resn) == len(resi):
                try:
                    begin_pos, end_pos = resi.index(begin_res), resi.index(end_res)
                    return ss[begin_pos:end_pos + 1], resi[begin_pos:end_pos + 1], resn[begin_pos:end_pos + 1]
                except ValueError:
                    print(f"residues_id {begin_res} or {end_res} not found in {ss_file_path}.")
            else:
                print(f"Mismatch in lengths of ss, resn, and resi in {ss_file_path}.")
    return None, None, None

