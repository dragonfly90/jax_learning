import os
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
import re

def convert_py_to_ipynb(py_file, ipynb_file):
    with open(py_file, 'r') as f:
        content = f.read()

    nb = nbf.v4.new_notebook()
    cells = []
    
    # Extract docstring if it exists at the top
    docstring_match = re.match(r'^"""(.*?)"""', content, re.DOTALL)
    if docstring_match:
        cells.append(nbf.v4.new_markdown_cell(docstring_match.group(1).strip()))
        content = content[docstring_match.end():].strip()
    
    # Split content into blocks based on headers
    # We use a regex that captures the separator so we can distinguish headers from code
    pattern = r'(#\s?----+.*?|#\s?\d+\.\s.*?)'
    parts = re.split(pattern, content)
    
    current_code = ""
    for part in parts:
        if not part: continue
        
        if re.match(pattern, part):
            # It's a header
            if current_code.strip():
                cells.append(nbf.v4.new_code_cell(current_code.strip()))
                current_code = ""
            # Convert '# ---- Header ----' to '## Header'
            header_text = part.strip('# -').strip()
            cells.append(nbf.v4.new_markdown_cell(f"## {header_text}"))
        else:
            # It's code
            current_code += part
    
    # Execute the notebook
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    try:
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(py_file)}})
    except Exception as e:
        print(f"Error executing {py_file}: {e}")
        
    with open(ipynb_file, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    scripts = sorted([f for f in os.listdir('.') if f.endswith('.py') and f[0].isdigit()])
    os.makedirs('notebooks', exist_ok=True)
    for script in scripts:
        print(f"Converting {script}...")
        convert_py_to_ipynb(script, f"notebooks/{script.replace('.py', '.ipynb')}")
