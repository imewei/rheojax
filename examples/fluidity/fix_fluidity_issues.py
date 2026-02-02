#!/usr/bin/env python
"""Fix common issues in fluidity notebooks."""
import json
from pathlib import Path


def add_compute_fit_quality_cell(nb: dict) -> bool:
    """Add compute_fit_quality function after imports cell if needed."""
    # Check if compute_fit_quality is already defined
    has_def = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if "def compute_fit_quality" in source:
                has_def = True
                break

    if has_def:
        return False  # Already has the function

    # Check if compute_fit_quality is used
    uses_func = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if "compute_fit_quality(" in source and "def compute_fit_quality" not in source:
                uses_func = True
                break

    if not uses_func:
        return False  # Doesn't need the function

    # Find the main imports cell (after colab setup) and insert new cell after
    insert_idx = None
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if "import matplotlib" in source or "import arviz" in source:
                insert_idx = i + 1
                break

    if insert_idx is None:
        insert_idx = 2  # Default position

    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def compute_fit_quality(y_true, y_pred):\n",
            '    """Compute R² and RMSE."""\n',
            "    y_true = np.asarray(y_true)\n",
            "    y_pred = np.asarray(y_pred)\n",
            "    residuals = y_true - y_pred\n",
            "    if y_true.ndim > 1:\n",
            "        residuals = residuals.ravel()\n",
            "        y_true = y_true.ravel()\n",
            "    ss_res = np.sum(residuals**2)\n",
            "    ss_tot = np.sum((y_true - np.mean(y_true))**2)\n",
            "    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0\n",
            "    rmse = np.sqrt(np.mean(residuals**2))\n",
            "    return {'R2': r2, 'RMSE': rmse}"
        ],
        "id": "compute-fit-quality-def"
    }

    nb["cells"].insert(insert_idx, new_cell)
    return True


def fix_variable_name_typos(nb: dict, filename: str) -> int:
    """Fix common variable name typos."""
    fixes = 0

    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            new_source = []
            for line in source:
                original = line
                # Fix t_data -> time where time is the actual variable
                if "t_data" in line and "time" not in line:
                    # Check if this is a definition or use
                    if "t_data =" not in line and "t_data=" not in line:
                        # It's a use, might need fixing depending on context
                        pass  # Keep as is for now, context-dependent
                # Fix stress/sigma/tau naming issues
                new_source.append(line)

            if new_source != source:
                cell["source"] = new_source
                fixes += 1

    return fixes


def ensure_outputs_dir_exists(nb: dict) -> bool:
    """Add os.makedirs for output directory if saving results."""
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if "output_dir" in source and "os.makedirs" not in source:
                # Need to check context
                pass
    return False


def fix_notebook(nb_path: Path) -> dict:
    """Fix issues in a single notebook."""
    try:
        with open(nb_path) as f:
            nb = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  ERROR: {nb_path.name} - Invalid JSON: {e}")
        return {"compute_fit_quality": False, "variable_fixes": 0, "output_dir": False, "error": True}

    changes = {
        "compute_fit_quality": False,
        "variable_fixes": 0,
        "output_dir": False,
        "error": False,
    }

    changes["compute_fit_quality"] = add_compute_fit_quality_cell(nb)
    changes["variable_fixes"] = fix_variable_name_typos(nb, nb_path.name)

    # Write back if changes were made
    if any(changes.values()):
        with open(nb_path, "w") as f:
            json.dump(nb, f, indent=1)

    return changes


def main():
    """Fix all fluidity notebooks."""
    fluidity_dir = Path(__file__).parent
    notebooks = sorted(fluidity_dir.glob("*.ipynb"))

    print(f"Found {len(notebooks)} notebooks")
    print("=" * 60)

    for nb_path in notebooks:
        changes = fix_notebook(nb_path)
        status = []
        if changes["compute_fit_quality"]:
            status.append("added compute_fit_quality")
        if changes["variable_fixes"]:
            status.append(f"fixed {changes['variable_fixes']} variables")
        if changes["output_dir"]:
            status.append("fixed output_dir")

        if status:
            print(f"{nb_path.name}: {', '.join(status)}")
        else:
            print(f"{nb_path.name}: no changes needed")

    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
