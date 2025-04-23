import json
import os

# Specify root directory containing subfolders with JSON files
ROOT_DIR = "/mnt/nsingh/open-unlearning/saves/testEvals_MUSE"  # Change this to your actual path

# Specify the keys to extract - TOFU
# KEYS_TO_EXTRACT = [
#     "forget_quality",
#     "model_utility",
#     "forget_truth_ratio",
#     "wf_Truth_Ratio"
# ]  # Modify this list as needed
#MUSE
KEYS_TO_EXTRACT = [
    "forget_knowmem_ROUGE",
    "retain_knowmem_ROUGE",
    "forget_ACCURACY",
    "retain_ACCURACY"
]  # Modify this list as needed

# Output LaTeX file
# OUTPUT_TEX_FILE = "./output_table.tex"

def extract_json_values(json_path, keys):
    """Extract specified key values from a JSON file."""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return [data.get(key, "N/A") for key in keys]
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return ["N/A"] * len(keys)

def generate_latex_table(data, keys):
    """Generate a LaTeX table from extracted data."""
    newkeys = []
    for k in keys:
        newkeys.append(k.split("_")[0])
    header = " & ".join(["Method"] + newkeys) + r" \\ \hline"
    subheader = "&  \multicolumn{2}{c}{Know-Mem-ROGUE}&  \multicolumn{2}{c}{Know-Mem-ACC}" + r" \\"
    rows = [
        " & ".join([folder] + [f"{val:.4f}" if isinstance(val, float) else str(val) for val in values]) + r" \\"
        for folder, values in data.items()
    ]
    
    latex_table = r"""
    \documentclass{article}
    \usepackage{booktabs}
    \begin{document}
    
    \begin{table}[h]
    \centering
    \begin{tabular}{""" + "l" + "c" * len(newkeys) + r"""}
    \toprule
    """ + subheader + r"""
    """ + header + r"""
    \midrule
    """ + "\n".join(rows) + r"""
    \bottomrule
    \end{tabular}
    \caption{\textbf{Some MUSE paraphrases.} Target model LLama-2-7B is tested, for both forget and retain set, we use one of the 10 perturbations.}
    \end{table}

    \end{document}
    """
    
    return latex_table

def main():
    """Main function to extract data and generate LaTeX table."""
    folder_data = {}

    # Traverse the root directory
    for folder in os.listdir(ROOT_DIR):
        for ev in ["evals_q_para1_Rpert","evals_q_para2_Rpert","evals_q_para3_Rpert","evals_q_para4_Rpert", "evals_q_para5_Rpert","evals_q_para6_Rpert", "evals_q_para7_Rpert","evals_q_para8_Rpert", "evals_question_Rpert"]: 
            folder_path = os.path.join(ROOT_DIR, folder) + f"/{ev}"

            if os.path.isdir(folder_path):
                json_files = [f for f in os.listdir(folder_path) if f.endswith("RY.json")]
                
                if json_files:
                    json_path = os.path.join(folder_path, json_files[0])  # Assume one JSON per folder
                    folder_data[folder + f"_{ev}"] = extract_json_values(json_path, KEYS_TO_EXTRACT)

    # Generate LaTeX table and save it
    latex_code = generate_latex_table(folder_data, KEYS_TO_EXTRACT)
    print(latex_code)
    # with open(OUTPUT_TEX_FILE, "w") as f:
    #     f.write(latex_code)
    
    # print(f"LaTeX table saved to {OUTPUT_TEX_FILE}")

if __name__ == "__main__":
    main()
