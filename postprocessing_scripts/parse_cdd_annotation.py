#!/usr/bin/python3
import re
from collections import defaultdict
import ast


# Define a function to process the string
def parse_cdddomain_annoation_string(domain_string):
    # Extract the domain name
    domain_name_match = re.match(r"domain: (\S+)_from_", domain_string)
    
    if domain_name_match:
        domain_name = domain_name_match.group(1)
        
        # Extract all start and end pairs
        start_end_pairs = re.findall(r"from_([\d,]+)_to_([\d,]+)", domain_string)
        
        # Prepare the result list
        result = []
        for start, end in start_end_pairs:
            # Split by commas and pair each corresponding start and end
            start_list = start.split(',')
            end_list = end.split(',')
            
            # Format each start-end pair
            for s, e in zip(start_list, end_list):
                result.append(f"{domain_name}:{s}-{e}")
        
        return result

    else:
        return []

# Function to extract numeric start and end values, ignoring any letters
def extract_start_end(value):
    # Extract the numeric part before and after the dash using regex
    start_end = re.findall(r'\d+', value.split(':')[1])
    start = int(start_end[0])
    end = int(start_end[1])
    return start, end


def parse_cdd_annotation(uniprotid, input_cdd_path):
    """
    This will read the uniprot cdd annotation files and make chain 
    """
    

    file_to_open = f"{input_cdd_path}/{uniprotid}_cdd_annotation.txt"
    

    with open(file_to_open, 'r') as file:
        uniprotid_cdd_domain = {}
        for line in file:
            line_contains = line.split()
            if "domains" in line_contains[1]:
                chain_id = line_contains[0]
                domains_str = ' '.join(line_contains[2:])
                #Convert the string to a list
                domains_str_list = ast.literal_eval(domains_str)
                chain_decompose = []
                for cdd_domain in domains_str_list:
                    chain_decompose.extend(parse_cdddomain_annoation_string(cdd_domain))
                sorted_chain_decompose = sorted(chain_decompose, key=lambda x: extract_start_end(x))
                if chain_id not in uniprotid_cdd_domain:
                    uniprotid_cdd_domain[chain_id] = sorted_chain_decompose
                else:
                    print(f"There is a problem with CDD domain annotation of id {uniprotid}")
        return uniprotid_cdd_domain 


if __name__ == "__main__":
    input_file_path = "../input/"
    unipro_id  = "P35613"
    cdd_domain = parse_cdd_annotation(unipro_id , f"{input_file_path}cdd_annotation/")
    print(cdd_domain)







