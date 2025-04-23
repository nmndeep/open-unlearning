from datasets import Dataset

# Load the .arrow file
arrow_file_path = "/mnt/nsingh/huggingface-models/huggingface/datasets/muse-bench___muse-news/knowmem/0.0.0/506bd5b150b92814d45e4404a82f120ab2d748bf/muse-news-forget_qa.arrow"
dataset = Dataset.from_file(arrow_file_path)
# table = dataset.to_table()

# Convert to a pandas DataFrame for easier handling
df = dataset.to_pandas()

# Select the first 10 rows and the desired columns
subset_df = df[['question', 'q_para4', 'q_para5', 'answer']].head(10)

# Begin LaTeX table code
latex_code = r"""\begin{table}[htbp]
\centering
\begin{tabular}{|p{3.5cm}|p{3.5cm}|p{3.5cm}|p{3.5cm}|}
\hline
\textbf{Question} & \textbf{Q\_Para1} & \textbf{Q\_Para3} & \textbf{Answer} \\
\hline
"""

# Add rows to LaTeX table, escaping special LaTeX characters
def escape_latex(s):
    if not isinstance(s, str):
        s = str(s)
    return s.replace('&', r'\&').replace('%', r'\%').replace('$', r'\$')\
            .replace('#', r'\#').replace('_', r'\_').replace('{', r'\{')\
            .replace('}', r'\}').replace('~', r'\textasciitilde{}')\
            .replace('^', r'\textasciicircum{}').replace('\\', r'\textbackslash{}')

for _, row in subset_df.iterrows():
    latex_code += f"{escape_latex(row['question'])} & {escape_latex(row['q_para4'])} & {escape_latex(row['q_para5'])} & {escape_latex(row['answer'])} \\\\\n\\hline\n"

# End LaTeX table code
latex_code += r"""\end{tabular}
\caption{Sample of Questions with Paragraphs and Answers}
\label{tab:sample_questions}
\end{table}
"""

# Print the LaTeX table code
print(latex_code)
