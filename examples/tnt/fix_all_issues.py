#!/usr/bin/env python3
"""Comprehensive fix for all TNT notebook API issues.

Issues fixed:
1. model.fit() returns self - fix .success, .params, .y_pred access
2. model.predict(oscillation) returns |G*| - use model.predict_saos() for (G', G'')
3. print_parameter_comparison expects (model, posterior_dict, param_names)
4. result_bayes.to_arviz() -> az.from_dict(posterior=...)
5. 'in result_bayes' membership checks
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

        # Fix 1: result_nlsq = model.fit(...) -> model.fit(...)
        if "result_nlsq = model.fit(" in source_str:
            source_str = source_str.replace("result_nlsq = model.fit(", "model.fit(")
            changes.append("Removed result_nlsq assignment")

        # Fix 2: result_nlsq.success
        if "result_nlsq.success" in source_str:
            source_str = re.sub(
                r'print\(f"\\n?NLSQ converged: \{result_nlsq\.success\}"\)',
                'print("NLSQ fit completed")',
                source_str,
            )
            source_str = source_str.replace("result_nlsq.success", "model.fitted_")
            changes.append("Fixed result_nlsq.success")

        # Fix 3: result_nlsq.params[name]
        if "result_nlsq.params[" in source_str:
            source_str = re.sub(
                r"result_nlsq\.params\[([^\]]+)\]",
                r"model.parameters.get_value(\1)",
                source_str,
            )
            changes.append("Fixed result_nlsq.params")

        # Fix 4: result_nlsq.y_pred
        if "result_nlsq.y_pred" in source_str:
            # Detect test_mode from context
            if "flow_curve" in source_str.lower():
                source_str = source_str.replace(
                    "result_nlsq.y_pred",
                    'model.predict(gamma_dot, test_mode="flow_curve")',
                )
            elif "oscillation" in source_str.lower() or "saos" in source_str.lower():
                source_str = source_str.replace(
                    "result_nlsq.y_pred",
                    'model.predict(omega, test_mode="oscillation")',
                )
            else:
                source_str = source_str.replace(
                    "result_nlsq.y_pred",
                    'model.predict(x, test_mode="flow_curve")',
                )
            changes.append("Fixed result_nlsq.y_pred")

        # Fix 5: G_prime, G_double_prime = model.predict(..., test_mode="oscillation")
        pattern = r'(G_prime[_\w]*),\s*(G_double_prime[_\w]*)\s*=\s*model\.predict\(([^,]+),\s*test_mode\s*=\s*["\']oscillation["\']\)'
        if re.search(pattern, source_str):
            source_str = re.sub(pattern, r"\1, \2 = model.predict_saos(\3)", source_str)
            changes.append("Changed predict(oscillation) to predict_saos()")

        # Fix 6: print_parameter_comparison first arg
        if "print_parameter_comparison(result_nlsq," in source_str:
            source_str = source_str.replace(
                "print_parameter_comparison(result_nlsq,",
                "print_parameter_comparison(model,",
            )
            changes.append("Fixed print_parameter_comparison first arg")

        # Fix 7: print_parameter_comparison second arg
        pattern = r"print_parameter_comparison\s*\(\s*model\s*,\s*result_bayes\s*,"
        if re.search(pattern, source_str):
            source_str = re.sub(
                pattern,
                "print_parameter_comparison(model, result_bayes.posterior_samples,",
                source_str,
            )
            changes.append("Fixed print_parameter_comparison second arg")

        # Fix 8: result_bayes.to_arviz()
        if "result_bayes.to_arviz()" in source_str:
            source_str = source_str.replace(
                "result_bayes.to_arviz()",
                "az.from_dict(posterior={name: result_bayes.posterior_samples[name][None, :] for name in param_names})",
            )
            changes.append("Fixed to_arviz() to az.from_dict()")

        # Fix 9: 'in result_bayes' membership check
        if re.search(r"\bin result_bayes\b(?!\.)", source_str):
            source_str = re.sub(
                r"\bin result_bayes\b(?!\.)",
                "in result_bayes.posterior_samples",
                source_str,
            )
            changes.append("Fixed 'in result_bayes'")

        # Fix 10: fit_metrics['nrmse'] -> fit_metrics['NRMSE']
        if "fit_metrics['nrmse']" in source_str:
            source_str = source_str.replace("fit_metrics['nrmse']", "fit_metrics['NRMSE']")
            changes.append("Fixed nrmse key case")

        if source_str != original_str:
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

    print(f"Fixing {len(notebooks)} TNT notebooks")
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
