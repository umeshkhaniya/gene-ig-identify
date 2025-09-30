import pandas as pd

input_data = "../src_strand/human_proteome_all.txt"

df = pd.read_csv(input_data, delim_whitespace=True)

df['igdomain_res_range'] = df['igdomain_res_range'].str.replace(':', '_')



df.to_excel("human_proteome_all_igstrand.xlsx", index=False)