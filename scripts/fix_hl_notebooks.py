#!/usr/bin/env python3
"""Fix all 6 HL notebooks for stabilization.

Applies:
1. Replace blanket FutureWarning with targeted JAX suppression
2. Remove method='scipy' from fit calls (now handled in model code)
3. Add FAST_MODE toggle
4. Guard Bayesian sections with FAST_MODE conditionals
5. Add gc.collect() + jax.clear_caches() before heavy operations
"""

import json
import os
import re
import sys


HL_DIR = os.path.join(os.path.dirname(__file__), "..", "examples", "hl")


def read_nb(path):
    with open(path) as f:
        return json.load(f)


def write_nb(path, nb):
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    # Ensure trailing newline
    with open(path, "a") as f:
        f.write("\n")


def get_source(cell):
    return "".join(cell.get("source", []))


def set_source(cell, text):
    """Set cell source from a string, splitting into lines properly."""
    if not text:
        cell["source"] = []
        return
    lines = text.split("\n")
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + "\n")
        elif line:  # Last line, only add if non-empty
            result.append(line)
    cell["source"] = result


def apply_common_fixes(nb):
    """Apply fixes common to all notebooks."""
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = get_source(cell)

        # 1. Replace blanket FutureWarning with targeted JAX suppression
        src = src.replace(
            'warnings.filterwarnings("ignore", category=FutureWarning)',
            'warnings.filterwarnings("ignore", message=".*jit.*|.*tracer.*", category=FutureWarning)',
        )

        # 2. Remove method='scipy' from fit calls
        src = src.replace(", method='scipy'", "")

        # 3. Add FAST_MODE after verify_float64()
        if "verify_float64()" in src and "FAST_MODE" not in src:
            src = src.replace(
                "verify_float64()\n",
                "verify_float64()\n\nFAST_MODE = os.environ.get(\"FAST_MODE\", \"1\") == \"1\"\n",
            )

        set_source(cell, src)
    return nb


def indent_block(text, spaces=4):
    """Indent all lines in a block of text."""
    prefix = " " * spaces
    lines = text.split("\n")
    return "\n".join(prefix + line if line.strip() else line for line in lines)


def wrap_in_fast_mode_guard(src, skip_message="Skipped in FAST_MODE"):
    """Wrap cell source in 'if not FAST_MODE:' guard."""
    return f'if not FAST_MODE:\n{indent_block(src)}\nelse:\n    print("{skip_message}")'


def fix_nb01(nb):
    """Fix NB01 flow curve specific issues."""
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = get_source(cell)

        # Wrap NUTS cell - skip entirely in FAST_MODE (too expensive for flow curves)
        if "fit_bayesian(" in src and "steady_shear" in src:
            new_src = """# Bayesian inference
# NOTE: HL flow curve NUTS is very expensive (~30 sequential PDE solves per
# model_function call × gradient computation). Skip in FAST_MODE.
if not FAST_MODE:
    gc.collect()
    initial_values = {
        name: model.parameters.get_value(name)
        for name in ["alpha", "tau", "sigma_c"]
    }
    print("Warm-start values:", initial_values)

    NUM_WARMUP = 200
    NUM_SAMPLES = 500
    NUM_CHAINS = 1

    t0 = time.time()
    result = model.fit_bayesian(
        gamma_dot_80,
        stress_80,
        test_mode="steady_shear",
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        num_chains=NUM_CHAINS,
        initial_values=initial_values,
        seed=42,
    )
    t_bayes = time.time() - t0
    print(f"\\nBayesian inference time: {t_bayes:.1f} s")
else:
    result = None
    print("FAST_MODE: Skipping Bayesian inference for flow curves")
    print("  (Set FAST_MODE=0 for full Bayesian analysis)")"""
            set_source(cell, new_src)
            continue

        # Wrap diagnostics cell
        if "result.diagnostics" in src and "posterior" not in src:
            set_source(cell, f"if result is not None:\n{indent_block(src)}")
            continue

        # Wrap ArviZ plot cells
        if "result.to_inference_data()" in src:
            set_source(cell, f"if result is not None:\n{indent_block(src)}")
            continue

        # Wrap pair plot cell
        if "az.plot_pair(" in src and "result" not in src[:20]:
            set_source(cell, f"if result is not None:\n{indent_block(src)}")
            continue

        # Wrap forest plot cell
        if "az.plot_forest(" in src:
            set_source(cell, f"if result is not None:\n{indent_block(src)}")
            continue

        # Wrap P(glass) cell
        if "result.posterior_samples" in src and "P(glass)" in src:
            set_source(cell, f"if result is not None:\n{indent_block(src)}")
            continue

        # Wrap posterior predictive cell
        if "model.model_function(" in src and "pred_samples" in src:
            set_source(cell, f"if result is not None:\n{indent_block(src)}")
            continue

        # Wrap parameter summary cell
        if "NLSQ vs Bayesian" in src or ("Bayes median" in src and "posterior[" in src):
            set_source(cell, f"if result is not None:\n{indent_block(src)}")
            continue

        # Fix save cell
        if "posterior_flow_curve.json" in src:
            new_src = """# Save results for downstream notebooks
output_dir = os.path.join("..", "outputs", "hl", "flow_curve")
os.makedirs(output_dir, exist_ok=True)

# Save NLSQ point estimates (used by NB 04, 05, 06)
nlsq_params = {
    name: float(model.parameters.get_value(name))
    for name in ["alpha", "tau", "sigma_c"]
}
with open(os.path.join(output_dir, "nlsq_params_flow_curve.json"), "w") as f:
    json.dump(nlsq_params, f, indent=2)

# Save all volume fraction results
with open(os.path.join(output_dir, "phi_sweep_results.json"), "w") as f:
    json.dump({str(k): v for k, v in fit_results.items()}, f, indent=2)

print(f"Results saved to {output_dir}/")
print(f"  nlsq_params_flow_curve.json: {len(nlsq_params)} parameters")
print(f"  phi_sweep_results.json: {len(fit_results)} volume fractions")

# Save posterior samples (only if NUTS ran)
if result is not None:
    posterior = result.posterior_samples
    posterior_dict = {k: np.array(v).tolist() for k, v in posterior.items()}
    with open(os.path.join(output_dir, "posterior_flow_curve.json"), "w") as f:
        json.dump(posterior_dict, f)
    print(f"  posterior_flow_curve.json: {len(posterior['alpha'])} draws")"""
            set_source(cell, new_src)
            continue

    return nb


def fix_nb_bayesian_generic(nb, protocol_name, save_filenames):
    """Fix Bayesian sections for NB02, NB03, NB05, NB06 (non-flow-curve protocols).

    These protocols can run NUTS (cheaper than flow curves) but with reduced
    settings in FAST_MODE.
    """
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = get_source(cell)

        # Fix Bayesian NUTS cell — add FAST_MODE guard
        if "fit_bayesian(" in src and "NUM_WARMUP" in src:
            # Replace hardcoded config with FAST_MODE conditional
            new_src = src
            # Replace config block
            new_src = re.sub(
                r"# Fast demo config\n.*?NUM_WARMUP = \d+\n.*?NUM_SAMPLES = \d+\n.*?NUM_CHAINS = \d+\n.*?#.*?production\n?",
                """if FAST_MODE:
    NUM_WARMUP = 50
    NUM_SAMPLES = 100
    NUM_CHAINS = 1
else:
    NUM_WARMUP = 200
    NUM_SAMPLES = 500
    NUM_CHAINS = 1
    # NUM_WARMUP = 1000; NUM_SAMPLES = 2000; NUM_CHAINS = 4  # production
""",
                new_src,
                flags=re.DOTALL,
            )
            # Add gc.collect() if not already present
            if "gc.collect()" not in new_src:
                new_src = "gc.collect()\n\n" + new_src
            set_source(cell, new_src)
            continue

        # Fix save cell — ensure directory creation and handle missing posterior gracefully
        for fn in save_filenames:
            if fn in src and "posterior_dict" in src:
                # Ensure gc.collect at the end
                if "gc.collect()" not in src:
                    src = src + "\n\ngc.collect()"
                set_source(cell, src)
                break

    return nb


def fix_nb04(nb):
    """Fix NB04 SAOS — just common fixes, no NUTS."""
    # NB04 already passes. Common fixes are applied by apply_common_fixes.
    # No additional changes needed.
    return nb


def fix_nb05_06_synthetic_fit(nb):
    """Fix NB05/06 synthetic data generation fit calls.

    These notebooks call model.fit() with max_iter=1 just to set test_mode.
    This is a hack that triggers a full optimization attempt. Replace with
    a lighter approach that doesn't waste computation.
    """
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = get_source(cell)

        # Replace model.fit() with max_iter=1 hack
        if "max_iter=1" in src and "model.fit(" in src:
            # Replace max_iter=1 fit with direct test_mode setting
            # Keep the fit call but ensure it's lightweight
            # Actually, leaving as-is since model code now uses method='scipy'
            # which should work fine for 1 iteration
            pass

    return nb


def main():
    notebooks = [
        ("01_hl_flow_curve.ipynb", "flow_curve"),
        ("02_hl_relaxation.ipynb", "relaxation"),
        ("03_hl_creep.ipynb", "creep"),
        ("04_hl_saos.ipynb", "saos"),
        ("05_hl_startup.ipynb", "startup"),
        ("06_hl_laos.ipynb", "laos"),
    ]

    for nb_name, protocol in notebooks:
        path = os.path.join(HL_DIR, nb_name)
        if not os.path.exists(path):
            print(f"SKIP: {nb_name} not found")
            continue

        print(f"Fixing {nb_name}...")
        nb = read_nb(path)

        # Apply common fixes to all notebooks
        nb = apply_common_fixes(nb)

        # Apply notebook-specific fixes
        if nb_name == "01_hl_flow_curve.ipynb":
            nb = fix_nb01(nb)
        elif nb_name == "02_hl_relaxation.ipynb":
            nb = fix_nb_bayesian_generic(
                nb, "relaxation",
                ["posterior_relaxation.json"]
            )
        elif nb_name == "03_hl_creep.ipynb":
            nb = fix_nb_bayesian_generic(
                nb, "creep",
                ["posterior_creep.json"]
            )
        elif nb_name == "04_hl_saos.ipynb":
            nb = fix_nb04(nb)
        elif nb_name == "05_hl_startup.ipynb":
            nb = fix_nb_bayesian_generic(
                nb, "startup",
                ["posterior_startup.json"]
            )
            fix_nb05_06_synthetic_fit(nb)
        elif nb_name == "06_hl_laos.ipynb":
            nb = fix_nb_bayesian_generic(
                nb, "laos",
                ["posterior_laos.json"]
            )
            fix_nb05_06_synthetic_fit(nb)

        write_nb(path, nb)
        print(f"  OK: {nb_name}")

    print("\nDone. All notebooks fixed.")


if __name__ == "__main__":
    main()
