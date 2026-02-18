#!/usr/bin/env python
"""Update TNT notebooks to use FAST_MODE configuration system."""

import json
import re
from pathlib import Path


def update_notebook(notebook_path: Path) -> tuple[bool, str]:
    """Update a single notebook to use FAST_MODE configuration.

    Returns:
        Tuple of (success, message).
    """
    with open(notebook_path) as f:
        nb = json.load(f)

    updated = False
    messages = []

    # Step 1: Update imports to include get_bayesian_config
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if "from tnt_tutorial_utils import" in source:
                # Check if get_bayesian_config is already imported
                if "get_bayesian_config" not in source:
                    # Find a good place to insert
                    for i, line in enumerate(cell["source"]):
                        # Look for common imports we can add after
                        for pattern in [
                            "print_parameter_comparison,",
                            "print_convergence_summary,",
                            "save_tnt_results,",
                            "get_tnt_",
                        ]:
                            if pattern in line and "get_bayesian_config" not in line:
                                # Insert get_bayesian_config after this line
                                cell["source"].insert(
                                    i + 1, "    get_bayesian_config,\n"
                                )
                                updated = True
                                messages.append("Added get_bayesian_config import")
                                break
                        if updated:
                            break
                break

    # Step 2: Update Bayesian configuration cell
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            # Look for the hardcoded NUM_WARMUP pattern
            if "NUM_WARMUP = " in source and "NUM_SAMPLES = " in source:
                # Check if already updated
                if "get_bayesian_config()" in source:
                    messages.append("Already updated")
                    continue

                # Build the replacement source
                # Extract any model-specific context from the existing cell
                # Find the fit_bayesian call to preserve its parameters
                fit_bayesian_match = re.search(
                    r"(\w+)\.fit_bayesian\s*\([^)]+\)", source, re.DOTALL
                )

                if fit_bayesian_match:
                    # Extract the model variable name
                    model_var = fit_bayesian_match.group(1)

                    # Extract test_mode and other kwargs from the call
                    call_text = fit_bayesian_match.group(0)
                    test_mode_match = re.search(
                        r'test_mode\s*=\s*["\'](\w+)["\']', call_text
                    )
                    test_mode = (
                        test_mode_match.group(1) if test_mode_match else "flow_curve"
                    )

                    # Build new cell based on test_mode
                    extra_kwargs = ""
                    if test_mode == "startup":
                        extra_kwargs = "\n    gamma_dot=gamma_dot,"
                    elif test_mode == "creep":
                        extra_kwargs = "\n    sigma_applied=sigma_applied,"
                    elif test_mode == "relaxation":
                        extra_kwargs = "\n    gamma_dot=gamma_dot,"
                    elif test_mode == "laos":
                        extra_kwargs = "\n    gamma_0=gamma_0,\n    omega=omega,"

                    new_source = f"""# Bayesian inference configuration (FAST_MODE controlled in tnt_tutorial_utils.py)
bayesian_config = get_bayesian_config()
NUM_WARMUP = bayesian_config["num_warmup"]
NUM_SAMPLES = bayesian_config["num_samples"]
NUM_CHAINS = bayesian_config["num_chains"]

# To override for production, set in tnt_tutorial_utils.py:
#   FAST_MODE = False  # Uses num_warmup=200, num_samples=500

print(f"Running NUTS with {{NUM_CHAINS}} chain(s)...")
print(f"Warmup: {{NUM_WARMUP}} samples, Sampling: {{NUM_SAMPLES}} samples")

start_time = time.time()
bayesian_result = {model_var}.fit_bayesian(
    x_data, y_data,
    test_mode="{test_mode}",{extra_kwargs}
    num_warmup=NUM_WARMUP,
    num_samples=NUM_SAMPLES,
    num_chains=NUM_CHAINS,
    seed=42
)
bayes_time = time.time() - start_time

print(f"\\nBayesian inference completed in {{bayes_time:.1f}} seconds")"""

                    cell["source"] = [new_source]
                    updated = True
                    messages.append(f"Updated Bayesian config for {test_mode} mode")

    # Write back if updated
    if updated:
        with open(notebook_path, "w") as f:
            json.dump(nb, f, indent=1)

    return updated, "; ".join(messages) if messages else "No changes needed"


def main():
    tnt_dir = Path("examples/tnt")
    notebooks = sorted(tnt_dir.glob("*.ipynb"))

    print(f"Found {len(notebooks)} notebooks")
    print("-" * 60)

    success_count = 0
    for nb_path in notebooks:
        updated, message = update_notebook(nb_path)
        status = "UPDATED" if updated else "SKIP"
        print(f"[{status}] {nb_path.name}: {message}")
        if updated:
            success_count += 1

    print("-" * 60)
    print(f"Updated {success_count}/{len(notebooks)} notebooks")


if __name__ == "__main__":
    main()
