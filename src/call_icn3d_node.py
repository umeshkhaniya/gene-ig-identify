#!/usr/bin/env python3
import pickle
import subprocess
import os
import gzip
import pandas as pd
import json
import requests



def check_filename_exist(file_name_tocheck, input_file_path):
    """
    This will check whether given file exits on the defined path.
    """
    if os.path.isfile(input_file_path+file_name_tocheck):
        return True
    else:
        return False

def download_structure(id_code, folder_path):
    """
    This will download the pdb files for you. id code can be either pdb id
    like 7urv or AF uniprot id: Q9UM44
    PDB file needed for node script to run.

    """
    stru_name_pdb = id_code.upper() +".pdb"
    stru_name_cif = id_code.upper() +".cif"



    if not check_filename_exist(stru_name_pdb, folder_path) and not check_filename_exist(stru_name_cif, folder_path):

        response = None

        if len(id_code) == 4:  # PDB ID (e.g., 4HEA)
            urls = [
                f"https://files.rcsb.org/download/{id_code}.pdb",
                f"https://files.rcsb.org/download/{id_code}.cif"
            ]
            
            for pdb_url in urls:
                response = requests.get(pdb_url, stream=True)
                if response.status_code == 200:
                    stru_name_pdb = os.path.basename(pdb_url)  # Set correct filename
                    break  # Stop at first successful download

        elif len(id_code) > 4:  # UniProt ID (e.g., Q9UM44)
            af_id = f"AF-{id_code}-F1-model_v4.pdb"
            url = f"https://alphafold.ebi.ac.uk/files/{af_id}"
            response = requests.get(url, stream=True)
            stru_name_pdb = af_id  # Set correct filename

        if response and response.status_code == 200:
            with open(os.path.join(folder_path, f"{id_code.upper()}.pdb"), 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded: {stru_name_pdb}")
        else:
            print(f"Failed to download structure for {id_code}")


def create_pdbss(stru_id, file_path):
    # make sure pdb
    file_to_save = stru_id.upper() + "_pdbdss.pkl.gz"
    pdb_path = file_path + "/pdb_files/"
    

    pdb_ss_path = file_path + "/pdb_ss/"

    if not check_filename_exist(file_to_save, pdb_ss_path):
        # add pdb file
        if check_filename_exist(stru_id.upper()+".pdb",pdb_path):
            command = ["node", "secondarystructure.js", pdb_path+stru_id.upper()+".pdb", "ss"]
            result = subprocess.run(command, capture_output=True, text=True)

            if result.stdout:

                try:
                   json_data = json.loads(result.stdout) # to save as json
                   with gzip.open(os.path.join(pdb_ss_path, file_to_save), 'wb') as f:
                        pickle.dump(json_data, f)
                        print(f"{file_to_save} is created.")
                except:
                    print(f"error")
            else:
                print(f"icn3d ss is not created for a {stru_id}")
    return

def create_icn3dss(stru_id, icn3d_ss_path):
    # make sure pdb
    file_to_save = stru_id.upper() + "_icn3dss.pkl.gz"

    #icn3d_ss_path = file_path + "/icn3dss/"

    if not check_filename_exist(file_to_save, icn3d_ss_path):
        command = ["node", "secondarystructure2.js", stru_id.upper()]
        result = subprocess.run(command, capture_output=True, text=True)

        if result.stdout:

            json_data = json.loads(result.stdout) # to save as json

            with gzip.open(os.path.join(icn3d_ss_path, file_to_save), 'wb') as f:
                pickle.dump(json_data, f)
                print(f"{file_to_save} is created.")
        else:
            print(f"icn3d ss is not created for a {stru_id}")
    return


def get_sequence_icn3d(stru_id, sequence_file_path):
    # make sure pdb
    file_to_save = stru_id.upper() + "_sequence.pkl.gz"

    #icn3d_ss_path = file_path + "/icn3dss/"

    if not check_filename_exist(file_to_save, sequence_file_path):
        command = ["node", "get_sequence_icn3d.js", stru_id.upper()]
        result = subprocess.run(command, capture_output=True, text=True)

        if result.stdout:
            json_data = json.loads(result.stdout) # to save as json

            with gzip.open(os.path.join(sequence_file_path, file_to_save), 'wb') as f:
                pickle.dump(json_data, f)
                print(f"{file_to_save} is created.")
        else:
            print(f"icn3d sequence file is not created for a {stru_id}")
    return

def diverse_covid(file_path):
    pdb_id_set = set()
    with open(file_path) as input_content:
        for line in input_content:
            line_strip = line.strip()
            if line_strip:
                pdb1, pdb2 = line_strip.split(" ")
                pdb_id_set.add(pdb1.split("_")[0].upper())
                pdb_id_set.add(pdb2.split("_")[0].upper())

    return pdb_id_set


def create_icn3dinteraction(stru_id, chain, icn3d_interaction_path):
    # make sure pdb
    file_to_save = f"{stru_id.upper()}_{chain}_icn3dinteraction.json"

    #icn3d_ss_path = file_path + "/icn3dss/"


    if not check_filename_exist(file_to_save, icn3d_interaction_path):
        command = ["node", "interactiondetail.js", stru_id.upper(), chain, chain, os.path.join(icn3d_interaction_path, file_to_save)]
        subprocess.run(command)
    return



if __name__ == "__main__":
    input_file_path = "../input"
    #pdbid = "1cd8"
    #pdbid = "Q9UM44"
    #input_data = "all_input_files.txt"

    #input_data = "../src_strand/input_dataset_remove_CD19_ORF_IgFn3like.xlsx"
    #input_data = "diverse_covid_pdb_set_chain.txt"
    #input_data = "../src_strand/human_proteome_all.txt"

    input_data = "../src_strand/human_unique_proteome_all_0.4.xlsx"

 

    #df = pd.read_csv(input_data, delim_whitespace=True)
    df = pd.read_excel(input_data)


    pdb_chain = set((x.split('_')[0], x.split('_')[1]) for x in df['pdbid_chain'])

    print(len(pdb_chain))

    #print(pdb_chain)
    
    
    #create_icn3dss(pdbid, input_file_path)

    #pdb_code = diverse_covid(input_data)
    #print(len(pdb_code))
   
    for pdbid, chain_id in pdb_chain:
        #print(pdbid, chain_id)
        # read the files and download pdb and icn3dss

        download_structure(pdbid, input_file_path+ "/pdb_files/")

        get_sequence_icn3d(pdbid, input_file_path+ "/sequence_file/")

        create_icn3dinteraction(pdbid, chain_id, input_file_path + "/icn3d_interactions/")
        create_icn3dss(pdbid, input_file_path + "/icn3dss/")
        #create_pdbss(pdbid, input_file_path)
    





