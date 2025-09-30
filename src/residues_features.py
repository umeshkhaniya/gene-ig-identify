#!/usr/bin/env python3

import numpy as np
import math

# This will extract the residues features based on the residues name.

aa_three_one = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "HSD": "H", "HSE":"H" }

# https://www.sciencedirect.com/topics/chemistry/amino-acid-residue
aa_polarity = {
        "E": "neg_charged", 
        "D": "neg_charged", 
        "K": "pos_charged",
        "R": "pos_charged",
        "H": "pos_charged",
        "S": "polar",
        "T": "polar",
        "Y": "polar",
        "N": "polar",
        "C": "polar",
        "Q": "polar",
        "G": "apolar",
        "A": "apolar",
        "V": "apolar",
        "L": "apolar",
        "I": "apolar",
        "M": "apolar",
        "P": "apolar",
        "F": "apolar",
        "W": "apolar"
        }

aa_aromicity = {"Y", "W", "F"}


aa_type_pos = {'A': 0, "C": 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 
               'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 
               'P': 12, 'Q': 13, 'R': 14, 'S': 15, 
               'T': 16, 'V': 17, 'W': 18, 'Y': 19} 

aa_prop_pos = {"neg_charged": 0, "pos_charged": 1, "polar": 2, "apolar": 3}
   


def onehot_aa_type(aa):
    """
    Input:
    aa: one letter residue code

    Output:
    size of one hot code numpy 1D array.

    """
    # Initialize a zero vector of the specified size
    size = len(aa_type_pos)

    one_hot_encoding = np.zeros(size, dtype=int)
    position = aa_type_pos.get(aa)  # Get the position for the residue
    
    if position is not None and position < size:
        one_hot_encoding[position] = 1  # Set the corresponding position to 1
    
    return one_hot_encoding

def onehot_aa_properties(aa):
    # Initialize a zero vector of the specified size

    one_hot_pro= np.zeros(4, dtype=int)
    aa_pro = aa_polarity.get(aa)  
    position_pro = aa_prop_pos[aa_pro]
    
    if position_pro is not None and position_pro < 4:
        one_hot_pro[position_pro] = 1  # Set the corresponding position to 1
    
    return one_hot_pro


def onehot_aromaticity2(aa):
    aromicity_one_hot =  np.zeros(2, dtype=int)
    if aa in aa_aromicity:
        aromicity_one_hot[0] = 1
    else:
        aromicity_one_hot[1] = 1
    return aromicity_one_hot

def is_cys(res):
    if res == "C":
        return np.array([1])
    else:
        return np.array([0])

def cys_onehot(aa1, aa2):
    """
    If two residues are cys.
    """
    cys_one_hot =  np.zeros(2, dtype=int)
    if aa1 in ("C") and aa2 in ("C"):
        cys_one_hot[0] = 1
    else:
        cys_one_hot[1] = 1
    return cys_one_hot







# source:EquiPNAS: https://academic.oup.com/nar/article/52/5/e27/7590918
#As the edge feature for the graph 
# we use the ratio of the logarithm of the absolute difference between the indices of the two residues 
#in the primary sequence and their Euclidean distance. 
 #The numerator of the ratio measures how far apart the two residues are in the primary sequence, 
#while the denominator measures their spatial distance in 3D space.

def optimized_sequence_distance(distance, res1_position, res2_position):
    # here res1_position and res2_position are different so. res1 can have letter so
    #
    index_difference = abs(res1_position - res2_position)
    log_index_diff = math.log(index_difference)
    ratio = log_index_diff / distance
    return ratio


def normalized_contact(contact_no, min_contact, max_contact):
    
    #Handle the case where min_len equals max_len

    if min_contact == max_contact:
        return 0.5 # if same.
    else:

        return (contact_no - min_contact) / (max_contact - min_contact)




if __name__ == "__main__":
    print(onehot_aa_type("D"))
    print(onehot_aromaticity2("Y"))
    print(onehot_aromaticity2("E"))
    print(onehot_aa_properties("D"))
    print(onehot_aa_properties("L"))
    print(optimized_sequence_distance(7, 10, 12))
    print(cys_onehot("C", "Y"))
    print(cys_onehot("C", "C"))
    # print(normalized_distance(5, contact_dict))




