import pandas as pd
from parse_cdd_annotation import parse_cdd_annotation

def extract_middle_position(res_range: str) -> int:
    """Extract the middle position from a residue range like 'A23_B138'."""
    start_str, end_str = res_range.split('_')
    start = int(''.join(filter(str.isdigit, start_str)))
    end = int(''.join(filter(str.isdigit, end_str)))
    return (start + end) // 2

def extract_reference_range(ref_range: str) -> tuple[int, int]:
    """Extract start and end integers from a reference range like 'Ig0_BSG1:23-138' from CDD annotation."""
    #print(ref_range)
    range_str = ref_range.split(':')[1]
    start, end = map(int, range_str.split('-'))
    return start, end

def middle_within_range(ig_range: str, ref_range: str) -> bool:
    """Check if the middle of ig_range falls within ref_range."""
    ref_start, ref_end = extract_reference_range(ref_range)
    
    ig_middle = extract_middle_position(ig_range)
  
    return ref_start <= ig_middle <= ref_end

def match_cdd_annotation(pdb_chain: str, ig_range: str, cdd_dir: str = "../input/cdd_annotation/") -> str | None:
    """Match a domain from CDD annotation file based on middle residue position."""
    try:
        pdb_id = pdb_chain.split("_")[0]
        cdd_data = parse_cdd_annotation(pdb_id, cdd_dir)

        if not cdd_data or pdb_chain not in cdd_data:
            return None

        for entry in cdd_data[pdb_chain]:
            #print(ig_range, entry)
            if middle_within_range(ig_range, entry):
                
                return entry.split(':')[0] # Return name part (e.g., 'Ig0_BSG1':ML:22-153)
        return None

    except (FileNotFoundError, KeyError, AttributeError):
        print(" CDD file Not found")
        return None

def add_cdd_annotation_column(df: pd.DataFrame, cdd_dir: str = "../../input/cdd_annotation/") -> pd.DataFrame:
    """Add 'cdd_annot' column to DataFrame using CDD annotations."""
    df['cdd_annotation'] = df.apply(
        lambda row: match_cdd_annotation(row['id_chain'], row['igdomain_res_range'], cdd_dir),
        axis=1
    )
    return df

# Example usage:
if __name__ == "__main__":
    input_file = "merged1_result_uniquegenes_human.xlsx"  # Or "output_tom_pdb_all_TM0.4.txt"
    df = pd.read_excel(input_file)
    df = add_cdd_annotation_column(df)
    df.to_excel("cdd_merged1_result_uniquegenes_human.xlsx", index=False)
