import json5, json
import os, sys 
import pandas as pd

def parse_uniprot(unipro_file_path):
    with open(unipro_file_path, 'r') as file:
        uniprot_data = file.read()
        uniprot_data = json.loads(uniprot_data)
    UniproParseInfo = {}
    for uniproinfo in uniprot_data:
        for unidata in uniprot_data[uniproinfo]:
            uniproid = unidata["primaryAccession"]
            uniprot_links = f"https://www.uniprot.org/uniprotkb/{uniproid}/entry"


            pdb_list = []
            af_list = []
            #print(unidata.keys())
            annotationScore = unidata.get("annotationScore")
            goannot = {"C":[], "P": [], "F": []}


         
            if "uniProtKBCrossReferences" in unidata:
                for databaseinfo in unidata["uniProtKBCrossReferences"]:
                    #print(databaseinfo['database'])
                    if databaseinfo['database'] == "PDB": #get chain
                        chains_value = next((item['value'] for item in databaseinfo["properties"] if item['key'] == 'Chains'), None)

                        if chains_value:
                            chain_id = chains_value.split('=')[0]
                        else:
                            chain_id = ""


                        pdb_list.append(databaseinfo['id']+"_"+ chain_id )

                    if databaseinfo['database'] == "AlphaFoldDB":
                        af_list.append(databaseinfo['id'])

                    if  databaseinfo['database'] == "GO":
                        for goterm in databaseinfo['properties']:
                            if goterm['key']== "GoTerm":
                                goaspect = goterm['value'].split(":",1) # first split ":"
                                aspect, term = goaspect[0], goaspect[1]

                                goannot[aspect].append(term)
               
            comment_text = []
            diseaserelated= ""
            if "comments" in unidata:
                for comment_info in unidata["comments"]:
                    #print(comment_info)
                    if "texts" in comment_info:
                        comment_text.append(comment_info["texts"][0]["value"])

                    if "commentType" in comment_info:
                        if "DISEASE" in comment_info["commentType"]:
                            #comment_info['disease']['diseaseId']

                            diseaserelated = "yes" 
            if "genes" in unidata:
                #print(uniproid,unidata["genes"])
                if "geneName" in  unidata["genes"][0]:
                    gene = unidata["genes"][0]["geneName"]["value"]
                elif "orfNames" in  unidata["genes"][0]:
                    
                    gene = unidata["genes"][0]["orfNames"][0]["value"]

                else:


                    print("Modify code to include following genes name:", unidata["genes"][0])
                    gene = "add_from_uni"

            else:
                gene = ""

            if "features" in unidata:
                uniprot_domain_resrange = []
                uniprot_domain_type = []
                for feature_data in unidata["features"]:
                    if feature_data["type"].lower() == "domain":
                        uniprot_domain_resrange.append(f"{feature_data['location']['start']['value']}:{feature_data['location']['end']['value']}")
                        uniprot_domain_type.append(feature_data["description"])


            uniprot_keyword = {}

            if "keywords" in unidata:
                for functinfo in unidata["keywords"]:
                    uniprot_keyword[functinfo["category"]] = uniprot_keyword.get(functinfo["category"], []) + [functinfo["name"]]
                    

            if uniproid not in UniproParseInfo:
                UniproParseInfo[uniproid] = {
                    "uniprot_domain_resrange": uniprot_domain_resrange,
                    "uniprot_domain_type": uniprot_domain_type,
                    "annotationScore":annotationScore,
                    "pdb": pdb_list,
                    "uniprot_gene": gene,# take first gene name
                    "cd_number": "",
                    "goannot_cell_component": goannot["C"],
                    "goannot_bio_process": goannot["P"],
                    "goannot_mol_function": goannot["F"],
                    "keyword_cell_component": uniprot_keyword.get("Cellular component",[]),
                    "keyword_bio_process":uniprot_keyword.get("Biological process",[]),
                    "keyword_mol_function":uniprot_keyword.get("Molecular function",[]),
                    "keyword_ligand":uniprot_keyword.get("Ligand",[]),
                    "keyword_coding_seq_diver":uniprot_keyword.get("Coding sequence diversity",[]),
                    "comments_text": comment_text,
                    "disease_related_uniprot" : diseaserelated,
                    "af_list": af_list,
                    "description": "",
                    "link": uniprot_links
                }

            if "proteinDescription" in unidata: # some of the uniprots ids are obsolute
                #print(unidata["proteinDescription"])
                if "recommendedName" in unidata["proteinDescription"]:
                    description = unidata["proteinDescription"]["recommendedName"]['fullName']["value"]
                    UniproParseInfo[uniproid]["description"] = description
                elif "submissionNames" in unidata["proteinDescription"]:
                    description = unidata["proteinDescription"]["submissionNames"][0]['fullName']["value"]
                    UniproParseInfo[uniproid]["description"] = description

                elif "alternativeNames" in unidata["proteinDescription"]:
                    description = unidata["proteinDescription"]["alternativeNames"][0]['fullName']["value"]
                    UniproParseInfo[uniproid]["description"] = description
                
                else:
                    UniproParseInfo[uniproid]["description"] = ""


                if "cdAntigenNames" in unidata["proteinDescription"]: # to get CD name
                    cdd_antigenname = unidata["proteinDescription"]["cdAntigenNames"][0]['value']
                    UniproParseInfo[uniproid]["cd_number"] = cdd_antigenname
                


                
            else:
                UniproParseInfo[uniproid]["description"] = ""
               


    return UniproParseInfo

if __name__ == "__main__":
    #uniprot_parse_data = parse_uniprot("../input/data_from_uniprot_browser/all_human_proteome_82493_april12_2024/uniprotkb_proteome_UP000005640_2024_04_12.json")
    uniprot_parse_data = parse_uniprot("../../../input/data_from_uniprot_browser/all_human_proteome_82493_april12_2024/sample1.json")

    print(uniprot_parse_data)
