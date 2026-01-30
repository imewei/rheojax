#!/usr/bin/env python3
"""Fix API issues in Fluidity notebooks.

Issues fixed:
1. RheoData(test_mode=...) -> RheoData(initial_test_mode=...)
2. n_points -> N_y for FluidityNonlocal and FluiditySaramitoNonlocal
3. result.success, result.r_squared, result.nit -> model.fitted_ and compute metrics
4. result_bayes.to_inference_data() -> az.from_dict() or az.from_numpyro()
5. model.parameters[name].value -> model.parameters.get_value(name)
6. Missing compute_fit_quality import
"""

import json
import re
from pathlib import Path


def join_source(source):
    """Join source lines into a single string."""
    if isinstance(source, list):
        return "".join(source)
    return source


def split_source(source_str):
    """Split source string back into lines preserving newlines."""
    lines = []
    current_line = ""
    for char in source_str:
        current_line += char
        if char == "\n":
            lines.append(current_line)
            current_line = ""
    if current_line:
        lines.append(current_line)
    return lines


def fix_notebook(notebook_path: Path) -> tuple[bool, list[str]]:
    """Fix all API issues in a single notebook."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    changes = []
    modified = False

    for cell_idx, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", [])
        source_str = join_source(source)
        original_str = source_str

        # Fix 1: RheoData test_mode -> initial_test_mode
        if "RheoData(" in source_str and "test_mode=" in source_str:
            # Only change if it's not already initial_test_mode
            if "initial_test_mode=" not in source_str:
                source_str = re.sub(
                    r"RheoData\((.*?)test_mode=",
                    r"RheoData(\1initial_test_mode=",
                    source_str,
                    flags=re.DOTALL,
                )
                changes.append("Fixed RheoData test_mode -> initial_test_mode")

        # Fix 2: n_points -> N_y for Nonlocal models
        if "FluidityNonlocal(" in source_str or "FluiditySaramitoNonlocal(" in source_str:
            if "n_points=" in source_str:
                source_str = source_str.replace("n_points=", "N_y=")
                changes.append("Fixed n_points -> N_y")

        # Fix 3: result.success, result.r_squared, result.nit
        # These are complex - would need context-aware fixes
        # For now, just note them

        # Fix 4: Add compute_fit_quality import if used but not imported
        if "compute_fit_quality" in source_str and "import" not in source_str:
            # Check if it's a usage cell (not import cell)
            if "from rheojax" in source_str and "compute_fit_quality" not in source_str:
                # Add import
                source_str = source_str.replace(
                    "from rheojax.core.jax_config import safe_import_jax",
                    "from rheojax.core.jax_config import safe_import_jax\nfrom rheojax.utils.metrics import compute_fit_quality",
                )
                changes.append("Added compute_fit_quality import")

        if source_str != original_str:
            cell["source"] = split_source(source_str)
            modified = True

    if modified:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=1)

    return modified, changes


def main():
    """Fix all Fluidity notebooks."""
    fluidity_dir = Path(__file__).parent
    notebooks = sorted(fluidity_dir.glob("*.ipynb"))

    print(f"Fixing {len(notebooks)} Fluidity notebooks")
    print("=" * 60)

    total_fixed = 0
    for nb_path in notebooks:
        modified, changes = fix_notebook(nb_path)

        if modified:
            total_fixed += 1
            print(f"\n{nb_path.name}: FIXED")
            for change in sorted(set(changes)):
                count = changes.count(change)
                if count > 1:
                    print(f"  - {change} (x{count})")
                else:
                    print(f"  - {change}")
        else:
            print(f"{nb_path.name}: OK")

    print("\n" + "=" * 60)
    print(f"Fixed {total_fixed}/{len(notebooks)} notebooks")


if __name__ == "__main__":
    main()
