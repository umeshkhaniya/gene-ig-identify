#!/usr/bin/env python3
import os, sys
import math
import numpy as np


"""
source: https://swift.cmbi.umcn.nl/gv/dssp/DSSP_2.html
H = α-helix
B = residue in isolated β-bridge
E = extended strand, participates in β ladder
G = 3-helix (310 helix)
I = 5 helix (π-helix)
T = hydrogen bonded turn
S = bend
"""

ss3_map = {
    'H': 'H',
    'B': 'C',
    'E': 'E',
    'G': 'H',
    'I': 'C',
    'T': 'C',
    'S': 'C',
    '-': 'C'
}

def onehot_ss3(ss):
    ss3_onehot =  np.zeros(3, dtype=int)

    if ss in ("H", "G"):
        ss3_onehot[0] = 1
    elif ss in ("E"):
        ss3_onehot[1] = 1

    else:
        ss3_onehot[2] = 1

    return ss3_onehot



# def onehot_phi_psi(phi, psi):
# 	return np.array([math.cos(math.radians(phi)), math.cos(math.radians(psi))])

# solvent accessibility
#https://academic.oup.com/nar/article/52/5/e27/7590918

# biopython accesibiity: 0 to 1
# three: 0-0.3, 0.3-0.6, 0.6-1

def onehot_sa3(sa):
    sa_one_hot =  np.zeros(3, dtype=int)
    if sa <= 0.3:
        sa_one_hot[0] = 1
    elif sa <= 0.6:
        sa_one_hot[1] = 1
    else:
        sa_one_hot[2] = 1
    
    return sa_one_hot




if __name__ == "__main__":
    print(onehot_sa3(1))

    print(onehot_sa3(0.6))
    print(onehot_sa3(0.2))

    print(onehot_phi_psi(-120, 30))
    print(onehot_ss3("H"))