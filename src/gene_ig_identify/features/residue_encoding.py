#!/usr/bin/env python3

import math
import numpy as np

aa_three_one = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "HSD": "H", "HSE": "H",
}

aa_polarity = {
    "E": "neg_charged", "D": "neg_charged", "K": "pos_charged", "R": "pos_charged",
    "H": "pos_charged", "S": "polar", "T": "polar", "Y": "polar", "N": "polar",
    "C": "polar", "Q": "polar", "G": "apolar", "A": "apolar", "V": "apolar",
    "L": "apolar", "I": "apolar", "M": "apolar", "P": "apolar", "F": "apolar", "W": "apolar",
}

aa_aromicity = {"Y", "W", "F"}
aa_type_pos = {"A": 0, "C": 1, "D": 2, "E": 3, "F": 4, "G": 5, "H": 6, "I": 7, "K": 8, "L": 9, "M": 10, "N": 11, "P": 12, "Q": 13, "R": 14, "S": 15, "T": 16, "V": 17, "W": 18, "Y": 19}
aa_prop_pos = {"neg_charged": 0, "pos_charged": 1, "polar": 2, "apolar": 3}


def onehot_aa_type(aa):
    size = len(aa_type_pos)
    one_hot_encoding = np.zeros(size, dtype=int)
    position = aa_type_pos.get(aa)
    if position is not None and position < size:
        one_hot_encoding[position] = 1
    return one_hot_encoding


def onehot_aa_properties(aa):
    one_hot_pro = np.zeros(4, dtype=int)
    aa_pro = aa_polarity.get(aa)
    position_pro = aa_prop_pos[aa_pro]
    if position_pro is not None and position_pro < 4:
        one_hot_pro[position_pro] = 1
    return one_hot_pro


def onehot_aromaticity2(aa):
    aromicity_one_hot = np.zeros(2, dtype=int)
    if aa in aa_aromicity:
        aromicity_one_hot[0] = 1
    else:
        aromicity_one_hot[1] = 1
    return aromicity_one_hot


def cys_onehot(aa1, aa2):
    cys_one_hot = np.zeros(2, dtype=int)
    if aa1 in ("C") and aa2 in ("C"):
        cys_one_hot[0] = 1
    else:
        cys_one_hot[1] = 1
    return cys_one_hot


def optimized_sequence_distance(distance, res1_position, res2_position):
    index_difference = abs(res1_position - res2_position)
    if index_difference <= 1:
        index_difference = 2
    return math.log(index_difference) / distance

