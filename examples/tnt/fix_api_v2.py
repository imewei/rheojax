#!/usr/bin/env python3
"""Fix API usage issues in TNT notebooks - robust version.

Handles source as list of lines properly.
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
    # Split but keep the newlines
    lines = []
    current_line = ""
    for char in source_str:
        current_line += char
        if char == "\n":
            lines.append(current_line)
            current_line = ""
    if current_line:  # Last line without newline
        lines.append(current_line)
    return lines


def fix_notebook(notebook_path: Path) -> tuple[bool, list[str]]:
    """Fix API issues in a single notebook."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    changes = []
    modified = False

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", [])
        # Join into single string for easier regex processing
        source_str = join_source(source)
        original_str = source_str

        # Fix 1: result_nlsq = model.fit(...) -> model.fit(...)
        # Also remove references to result_nlsq.success and result_nlsq.y_pred
        if "result_nlsq = model.fit(" in source_str:
            source_str = source_str.replace("result_nlsq = model.fit(", "model.fit(")
            changes.append("Removed result_nlsq assignment")

        # Fix 2: result_nlsq.success -> True (or remove the line)
        if "result_nlsq.success" in source_str:
            # Replace the whole print statement
            source_str = re.sub(
                r'print\(f"\\n?NLSQ converged: \{result_nlsq\.success\}"\)',
                'print("\\nNLSQ fit completed (check model.fitted_)")',
                source_str,
            )
            source_str = source_str.replace("result_nlsq.success", "model.fitted_")
            changes.append("Fixed result_nlsq.success")

        # Fix 3: result_nlsq.params[name] -> model.parameters.get_value(name)
        if "result_nlsq.params[" in source_str:
            source_str = re.sub(
                r"result_nlsq\.params\[([^\]]+)\]",
                r"model.parameters.get_value(\1)",
                source_str,
            )
            changes.append("Fixed result_nlsq.params access")

        # Fix 4: result_nlsq.y_pred -> model.predict(x, test_mode=...)
        # This is tricky - replace with model.predict(gamma_dot, test_mode="flow_curve")
        if "result_nlsq.y_pred" in source_str:
            # For flow_curve notebooks
            source_str = source_str.replace(
                "result_nlsq.y_pred",
                'model.predict(gamma_dot, test_mode="flow_curve")',
            )
            changes.append("Fixed result_nlsq.y_pred")

        # Fix 5: G_prime..., G_double_prime... = model.predict(..., test_mode="oscillation")
        # -> model.predict_saos(omega)
        pattern = r'(G_prime[_\w]*),\s*(G_double_prime[_\w]*)\s*=\s*model\.predict\(([^,]+),\s*test_mode\s*=\s*["\']oscillation["\']\)'
        if re.search(pattern, source_str):
            source_str = re.sub(
                pattern, r"\1, \2 = model.predict_saos(\3)", source_str
            )
            changes.append("Changed model.predict(oscillation) to model.predict_saos()")

        # Fix 6: print_parameter_comparison(result_nlsq, result_bayes, param_names)
        # The first arg should be model, not result_nlsq
        if "print_parameter_comparison(result_nlsq," in source_str:
            source_str = source_str.replace(
                "print_parameter_comparison(result_nlsq,",
                "print_parameter_comparison(model,",
            )
            changes.append("Fixed print_parameter_comparison first arg")

        # Fix 7: print_parameter_comparison(model, result_bayes, param_names)
        # Second arg should be result_bayes.posterior_samples
        if re.search(
            r"print_parameter_comparison\s*\(\s*model\s*,\s*result_bayes\s*,", source_str
        ):
            source_str = re.sub(
                r"print_parameter_comparison\s*\(\s*model\s*,\s*result_bayes\s*,",
                "print_parameter_comparison(model, result_bayes.posterior_samples,",
                source_str,
            )
            changes.append("Fixed print_parameter_comparison second arg")

        # Fix 8: 'in result_bayes' -> 'in result_bayes.posterior_samples'
        if re.search(r"\bin result_bayes\b(?!\.)", source_str):
            source_str = re.sub(
                r"\bin result_bayes\b(?!\.)",
                "in result_bayes.posterior_samples",
                source_str,
            )
            changes.append("Fixed 'in result_bayes'")

        if source_str != original_str:
            # Convert back to list format
            cell["source"] = split_source(source_str)
            modified = True

    if modified:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=1)

    return modified, changes


def main():
    """Fix all TNT notebooks."""
    tnt_dir = Path(__file__).parent
    notebooks = sorted(tnt_dir.glob("*.ipynb"))

    print(f"Found {len(notebooks)} TNT notebooks")
    print("=" * 60)

    total_fixed = 0
    for nb_path in notebooks:
        modified, changes = fix_notebook(nb_path)

        if modified:
            total_fixed += 1
            print(f"\n{nb_path.name}: FIXED")
            for change in set(changes):  # Deduplicate
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
