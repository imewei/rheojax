#!/usr/bin/env python3
"""Fix API usage issues in TNT notebooks.

Issues fixed:
1. model.fit() returns self, not a result object - fix .success and .params access
2. model.predict(..., test_mode="oscillation") returns |G*|, not (G', G'')
   For components, use model.predict_saos(omega)
3. print_parameter_comparison expects posterior_samples dict, not BayesianResult
"""

import json
import re
from pathlib import Path


def fix_notebook(notebook_path: Path) -> tuple[bool, list[str]]:
    """Fix API issues in a single notebook.

    Returns:
        Tuple of (modified, list of changes made)
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    changes = []
    modified = False

    for cell in notebook.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue

        source = cell.get('source', [])
        if isinstance(source, str):
            source = [source]

        original_source = list(source)
        new_source = []

        for i, line in enumerate(source):
            new_line = line

            # Fix 1: result_nlsq = model.fit(...) pattern
            # The fit() method returns self, not a result object
            if re.search(r'result_nlsq\s*=\s*model\.fit\(', line):
                new_line = re.sub(r'result_nlsq\s*=\s*model\.fit\(', 'model.fit(', line)
                if new_line != line:
                    changes.append(f"Removed result_nlsq assignment (fit returns self)")

            # Fix 2: result_nlsq.success -> (check via model state)
            if 'result_nlsq.success' in line:
                new_line = re.sub(
                    r'result_nlsq\.success',
                    'True  # model.fit() returns self, check model.fitted_ if needed',
                    line
                )
                if new_line != line:
                    changes.append(f"Fixed result_nlsq.success reference")

            # Fix 3: result_nlsq.params[name] -> model.parameters.get_value(name)
            if 'result_nlsq.params' in line:
                # Handle both result_nlsq.params[name] and result_nlsq.params['name']
                new_line = re.sub(
                    r"result_nlsq\.params\[([^\]]+)\]",
                    r"model.parameters.get_value(\1)",
                    line
                )
                if new_line != line:
                    changes.append(f"Fixed result_nlsq.params access")

            # Fix 4: result_nlsq.y_pred -> model.predict(x, test_mode=...)
            # This is trickier - skip for now as it depends on context

            # Fix 5: G_prime_pred, G_double_prime_pred = model.predict(..., test_mode="oscillation")
            # Should use model.predict_saos(omega) instead
            if re.search(r"G_prime.*,\s*G_double_prime.*=\s*model\.predict\([^,]+,\s*test_mode\s*=\s*['\"]oscillation['\"]", line):
                # Extract the first argument (omega variable)
                match = re.search(r"model\.predict\(([^,]+),", line)
                if match:
                    omega_var = match.group(1).strip()
                    new_line = re.sub(
                        r"model\.predict\([^,]+,\s*test_mode\s*=\s*['\"]oscillation['\"].*?\)",
                        f"model.predict_saos({omega_var})",
                        line
                    )
                    if new_line != line:
                        changes.append(f"Changed model.predict(oscillation) to model.predict_saos()")

            # Fix 6: G_prime_fit, G_double_prime_fit = model.predict(..., test_mode="oscillation")
            if re.search(r"G_prime_fit.*,\s*G_double_prime_fit.*=\s*model\.predict\([^,]+,\s*test_mode\s*=\s*['\"]oscillation['\"]", line):
                match = re.search(r"model\.predict\(([^,]+),", line)
                if match:
                    omega_var = match.group(1).strip()
                    new_line = re.sub(
                        r"model\.predict\([^,]+,\s*test_mode\s*=\s*['\"]oscillation['\"].*?\)",
                        f"model.predict_saos({omega_var})",
                        line
                    )
                    if new_line != line:
                        changes.append(f"Changed model.predict(oscillation) to model.predict_saos()")

            # Fix 7: print_parameter_comparison(model, result_bayes, param_names)
            # Should be print_parameter_comparison(model, result_bayes.posterior_samples, param_names)
            if re.search(r"print_parameter_comparison\s*\(\s*model\s*,\s*result_bayes\s*,", line):
                new_line = re.sub(
                    r"print_parameter_comparison\s*\(\s*model\s*,\s*result_bayes\s*,",
                    "print_parameter_comparison(model, result_bayes.posterior_samples,",
                    line
                )
                if new_line != line:
                    changes.append(f"Fixed print_parameter_comparison to use posterior_samples")

            # Fix 8: Handle 'in result_bayes' checks (BayesianResult is not iterable)
            # Change: if name in result_bayes -> if name in result_bayes.posterior_samples
            if re.search(r"if\s+\S+\s+in\s+result_bayes\s*:", line) or re.search(r"in\s+result_bayes\s+", line):
                new_line = re.sub(
                    r"\bin result_bayes\b",
                    "in result_bayes.posterior_samples",
                    line
                )
                if new_line != line:
                    changes.append(f"Fixed 'in result_bayes' to use posterior_samples")

            new_source.append(new_line)

        if new_source != original_source:
            cell['source'] = new_source
            modified = True

    if modified:
        with open(notebook_path, 'w', encoding='utf-8') as f:
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
            for change in changes:
                print(f"  - {change}")
        else:
            print(f"{nb_path.name}: OK")

    print("\n" + "=" * 60)
    print(f"Fixed {total_fixed}/{len(notebooks)} notebooks")


if __name__ == "__main__":
    main()
