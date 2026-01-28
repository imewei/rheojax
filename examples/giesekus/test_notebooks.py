import os
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

notebooks_dir = Path('/Users/b80985/Projects/rheojax/examples/giesekus')
results = []

for nb_name in sorted(notebooks_dir.glob('*.ipynb')):
    if 'archive' in str(nb_name) or 'checkpoint' in str(nb_name):
        continue

    original_cwd = os.getcwd()
    try:
        os.chdir(nb_name.parent)
        nb = nbformat.read(nb_name.name, as_version=4)
        client = NotebookClient(nb, timeout=180, kernel_name='python3')
        client.execute()
        print(f'{nb_name.name}: ✓')
        results.append((nb_name.name, 'PASS', None))
    except CellExecutionError as e:
        error_msg = str(e)[:200]
        print(f'{nb_name.name}: ✗')
        print(f'  Error: {error_msg}')
        results.append((nb_name.name, 'FAIL', error_msg))
    except Exception as e:
        error_msg = str(e)[:200]
        print(f'{nb_name.name}: ✗ (unexpected error)')
        print(f'  Error: {error_msg}')
        results.append((nb_name.name, 'ERROR', error_msg))
    finally:
        os.chdir(original_cwd)

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
for name, status, error in results:
    if status == 'PASS':
        print(f"✓ {name}")
    else:
        print(f"✗ {name}")
        if error:
            print(f"  {error[:100]}")
