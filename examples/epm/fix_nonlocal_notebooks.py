"""Fix FluiditySaramitoNonlocal notebooks API issues.

This script fixes the following common issues across notebooks 19-24:
1. Replace n_points with N_y
2. Replace gap_width with H
3. Remove utility imports that don't exist
4. Fix parameter setting methods
5. Fix method signatures
"""

import json
from pathlib import Path
import re


def fix_notebook(notebook_path: Path) -> None:
    """Fix a single notebook."""
    print(f"Fixing {notebook_path.name}...")

    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    modified = False

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            original = source

            # Fix 1: Replace n_points with N_y
            source = source.replace('n_points=51', 'N_y=51')
            source = source.replace('n_points=31', 'N_y=31')
            source = source.replace('n_points=', 'N_y=')
            source = source.replace('model.n_points', 'model.N_y')
            source = source.replace('model_true.n_points', 'model_true.N_y')
            source = source.replace('model_fit.n_points', 'model_fit.N_y')

            # Fix 2: Replace gap_width with H
            source = source.replace('gap_width=1e-3', 'H=1e-3')
            source = source.replace('gap_width=', 'H=')
            source = source.replace('model.gap_width', 'model.H')
            source = source.replace('model_true.gap_width', 'model_true.H')
            source = source.replace('model_fit.gap_width', 'model_fit.H')
            source = source.replace("params['gap_width']", "params['H']")
            source = source.replace('"gap_width":', '"H":')
            source = source.replace("'gap_width':", "'H':")

            # Fix 3: Remove utility imports
            lines = source.split('\n')
            new_lines = []
            skip_next = False

            for i, line in enumerate(lines):
                # Skip utility imports
                if 'from fluidity_tutorial_utils import' in line or \
                   'import fluidity_tutorial_utils' in line or \
                   'utils_path = Path' in line or \
                   'sys.path.insert' in line and 'utils' in line:
                    continue

                # Skip utility function calls
                if any(func in line for func in [
                    'get_output_dir',
                    'save_fluidity_results',
                    'print_convergence_summary',
                    'print_parameter_comparison',
                    'get_fluidity_param_names',
                    'detect_shear_banding'
                ]):
                    # Replace with inline equivalent
                    if 'get_output_dir' in line:
                        new_lines.append(line.replace('get_output_dir', 'Path'))
                        continue
                    elif 'detect_shear_banding' in line:
                        new_lines.append('# Banding detection (inline)')
                        continue
                    else:
                        continue

                new_lines.append(line)

            source = '\n'.join(new_lines)

            # Fix 4: Replace set_values() calls
            # Pattern: model.parameters.set_values({...})
            source = re.sub(
                r'model(_\w+)?\.parameters\.set_values\(\s*\{([^}]+)\}\s*\)',
                lambda m: '\n'.join(
                    f'{m.group(1) or "model"}.parameters.{key.strip()}.value = {val.strip()}'
                    for line in m.group(2).split(',')
                    if ':' in line
                    for key, val in [line.split(':', 1)]
                ),
                source
            )

            # Fix 5: Fix detect_shear_bands method signature
            source = source.replace(
                'model.detect_shear_bands(f_field_final, threshold=0.3)',
                'model.detect_shear_bands(f_field_final)'
            )
            source = source.replace(
                'detect_shear_banding(f_field, threshold=0.1)',
                'detect_shear_banding(f_field)'
            )

            # Fix 6: Replace RheoData creation patterns
            source = source.replace(
                'rheo_data = RheoData(\n    x=t,\n    y=sigma_bulk_noisy,\n    test_mode=\'startup\'\n)',
                'rheo_data = RheoData(\n    x=t,\n    y=sigma_bulk_noisy,\n    test_mode=\'startup\',\n    x_label=\'time\',\n    y_label=\'stress\'\n)'
            )

            # Fix 7: Replace model_fit.parameters.get_value() if it's wrong API
            source = source.replace('.parameters.get_value(', '.parameters[')
            source = source.replace('.get_value(', '.parameters[')

            if source != original:
                cell['source'] = source.split('\n')
                if not cell['source'][-1].endswith('\n'):
                    cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
                modified = True

    if modified:
        # Backup original
        backup_path = notebook_path.with_suffix('.ipynb.bak')
        if not backup_path.exists():
            notebook_path.rename(backup_path)
            print(f"  Backed up to {backup_path.name}")

        # Save fixed version
        with open(notebook_path, 'w') as f:
            json.dump(nb, f, indent=1)
        print(f"  ✓ Fixed and saved")
    else:
        print(f"  No changes needed")


def main():
    """Fix all nonlocal notebooks."""
    notebooks_dir = Path(__file__).parent.parent / 'fluidity'

    notebook_files = [
        '19_saramito_nonlocal_flow_curve.ipynb',
        '20_saramito_nonlocal_startup.ipynb',
        '21_saramito_nonlocal_creep.ipynb',
        '22_saramito_nonlocal_relaxation.ipynb',
        '23_saramito_nonlocal_saos.ipynb',
        '24_saramito_nonlocal_laos.ipynb',
    ]

    print("Fixing FluiditySaramitoNonlocal notebooks...")
    print("=" * 60)

    for notebook_name in notebook_files:
        notebook_path = notebooks_dir / notebook_name
        if notebook_path.exists():
            fix_notebook(notebook_path)
        else:
            print(f"⚠ {notebook_name} not found")

    print("\n" + "=" * 60)
    print("✓ All notebooks processed")
    print("\nBackup files (.ipynb.bak) created for safety")
    print("Test the notebooks and remove backups if everything works")


if __name__ == '__main__':
    main()
