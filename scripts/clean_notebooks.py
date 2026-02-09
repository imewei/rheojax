#!/usr/bin/env python3
"""
Script to clean outputs and execution counts from Jupyter notebooks in the examples directory.
Usage: python scripts/clean_notebooks.py
"""

import os
import nbformat

def clean_notebook(file_path):
    """
    Reads a notebook, clears all outputs and execution counts, and saves it back.
    """
    try:
        # Read the notebook
        with open(file_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Iterate over cells and clear outputs/execution_count
        cleaned = False
        for cell in nb.cells:
            if cell.cell_type == 'code':
                if cell.get('outputs') or cell.get('execution_count'):
                    cell['outputs'] = []
                    cell['execution_count'] = None
                    cleaned = True
        
        # Save back if potential changes (or just to enforce cleanliness)
        if cleaned:
            with open(file_path, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)
            print(f"Cleaned: {file_path}")
        else:
            print(f"Already clean: {file_path}")
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    base_dir = os.path.join(os.getcwd(), 'examples')
    print(f"Scanning for notebooks in: {base_dir}")
    
    notebook_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.ipynb') and not file.startswith('.'):
                notebook_files.append(os.path.join(root, file))
    
    print(f"Found {len(notebook_files)} notebooks. Cleaning...")
    
    for file_path in notebook_files:
        clean_notebook(file_path)
        
    print("Done.")

if __name__ == "__main__":
    main()
