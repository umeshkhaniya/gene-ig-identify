import subprocess
import multiprocessing
import os
import json5, json
import time
import pandas as pd



def read_uniprot_id(uni_filename):
    # order important so. Make list
    uniprots_id = list()
    with open(uni_filename) as input_content:
        for line in input_content:
            uniprots_id.append(line.strip())
           
    return uniprots_id

def read_text_file(ids_file):
    df = pd.read_csv(ids_file, delim_whitespace=True)
    # Extract PDB ID and chain ID
    df[["pdb", "chainid"]] = df["pdbid_chain"].str.split("_", expand=True)

    pdb_ids = set(df['pdb'])
    return list(pdb_ids)

def read_excel_file(ids_file):
    df = pd.read_excel(ids_file)
    # Extract PDB ID and chain ID
    df[["id", "chain"]] = df["id_chain"].str.split("_", expand=True)

    pdb_ids = set(df['id'])
    return list(pdb_ids)



def process_batch_onepdb(pdb_chunk):
    """
    This will create the mapping  numbering file by calling tm align to
    get the mappping files.
    """
    for one_pdb in pdb_chunk:
        print(f"process: {one_pdb}")
        output_file = f"{one_pdb}_cdd_annotation.txt"
        command = ["node", "cdd_annotation.js", one_pdb, "3"] # 3 for conserved domain
        result = subprocess.run(command, capture_output=True, text=True)

    # if error then don't run again that files. save that ids in json. 
    #Write out already created files so that we don't do future running again.
        result_ig_parse = result.stdout
        if len(result_ig_parse) > 3:
            with open(output_path+output_file, 'w') as f:
                    f.write(result_ig_parse) 
                    print(f"{output_file} is created in {output_path}{output_file}.")
        else:
            if os.path.exists(output_path + unipro_nostructure_ids):
                mode = 'a'  # Append mode
            else:
                mode = 'w'  # Write mode

            # Write out the result
            with open(output_path + unipro_nostructure_ids, mode) as f:
                f.write(one_pdb+'\n')
            # with open(output_path+output_file, 'w') as f:
            #         f.write(result_ig_parse) 
            #         print(f"{output_file} is created in {output_path}{output_file}.")
                
# let run until it makes all the pdb files.

def remove_elements_in_set(input_process_list, remove_set):
    # Create a copy of the list to avoid modifying the original list
    updated_list = input_process_list[:]
    # Iterate over the list and remove elements if they are present in the set
    for item in input_process_list:
        if item in remove_set:
            updated_list.remove(item)
    # Return the updated list
    return updated_list     


    

output_path = "../input/cdd_annotation/" # here I do for structures
unipro_nostructure_ids = "notproceesed_ids.txt"


def main():
    
    #input_list = read_uniprot_id("../input/human_uni_pro_all.txt")
    #input_list  = read_uniprot_id("../input/unique_genes_uniprotids.txt")
    #input_list  = read_file("./human_proteome_all_TM0.4.txt")
    #input_list  = read_uniprot_id("./unique_genes_uniprotids.txt")
    #input_list  = read_text_file("./output_tom_pdb_all_TM0.4.txt")
    input_list = read_excel_file("./unique_genes/human_unique_proteome_all_0.4.xlsx")

    print(f"Total input ids: {len(input_list)}")

    json_file_precessed_ids = {filename.split(".")[0].split("_")[0] for filename in os.listdir(output_path) if filename.endswith('.txt')}

    # read if exits.

    if os.path.isfile(f"{output_path}/notproceesed_ids.txt"):
        non_processed_list = set(read_uniprot_id(f"{output_path}/notproceesed_ids.txt")) 
        print(f"Total processed problem: {len(non_processed_list)}")

        
    else:
        non_processed_list = set()
        


    print(f"Total ids processed: {len(json_file_precessed_ids)}")


    # run maximum time of defined max_iteration or until list remains

        # Number of CPU cores
    remove_process_ids = remove_elements_in_set(input_list, json_file_precessed_ids)
    remaining_process_ids = remove_elements_in_set(remove_process_ids, non_processed_list)
    print(f"Remainings ids to process: {len(remaining_process_ids)}")

   
    
    process_batch_onepdb(remaining_process_ids)
        

if __name__ == "__main__":
    main()


