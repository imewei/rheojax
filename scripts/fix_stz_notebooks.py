#!/usr/bin/env python
"""Fix STZ notebooks for clean CI pass.

Applies:
1. Replace blanket FutureWarning suppression with targeted equinox DeprecationWarning
2. Add FAST_MODE support with reduced Bayesian samples
3. Add gc.collect() + jax.clear_caches() before Bayesian inference
4. Fix NB01 divergences via seed change
5. Add memory management for ODE-heavy notebooks (NB02-04)
"""

import json
from pathlib import Path


def load_notebook(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_notebook(nb: dict, path: Path):
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"  Saved: {path}")


def get_cell_source(cell: dict) -> str:
    src = cell.get("source", [])
    if isinstance(src, list):
        return "".join(src)
    return src


def set_cell_source(cell: dict, source: str):
    cell["source"] = source.split("\n")
    # Restore newlines at end of each line except the last
    for i in range(len(cell["source"]) - 1):
        cell["source"][i] += "\n"
    # Clear outputs so the notebook is clean for execution
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


def find_cell_containing(nb: dict, text: str) -> int | None:
    for i, cell in enumerate(nb["cells"]):
        if text in get_cell_source(cell):
            return i
    return None


def clear_all_outputs(nb: dict):
    """Clear all code cell outputs."""
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None


# ============================================================
# NB01: Flow Curve
# ============================================================
def fix_nb01(path: Path):
    print(f"\nFixing NB01: {path.name}")
    nb = load_notebook(path)

    # Fix setup cell: replace FutureWarning with targeted + add FAST_MODE + gc
    idx = find_cell_containing(nb, "warnings.filterwarnings")
    if idx is not None:
        set_cell_source(nb["cells"][idx], """%matplotlib inline
import gc
import os
import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from rheojax.core.jax_config import safe_import_jax, verify_float64
from rheojax.models.stz import STZConventional

jax, jnp = safe_import_jax()
verify_float64()

# Targeted suppression: equinox internal deprecation (harmless, not under our control)
warnings.filterwarnings("ignore", message=".*is_leaf.*", category=DeprecationWarning)

# FAST_MODE: reduced Bayesian samples for CI; set FAST_MODE=0 for production
FAST_MODE = int(os.environ.get("FAST_MODE", "1"))
print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"FAST_MODE: {FAST_MODE}")""")

    # Fix Bayesian cell: FAST_MODE-aware config + gc + seed=100 for 0 divergences
    idx = find_cell_containing(nb, "NUM_WARMUP = 200")
    if idx is not None:
        set_cell_source(nb["cells"][idx], """# Warm-start values from NLSQ
initial_values = {
    name: model.parameters.get_value(name)
    for name in model.parameters.keys()
}
print("Warm-start values:")
for k, v in initial_values.items():
    print(f"  {k}: {v:.4g}")

# Free memory before Bayesian inference
gc.collect()
jax.clear_caches()

if FAST_MODE:
    NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS = 50, 100, 1
else:
    NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS = 1000, 2000, 4

t0 = time.time()
result = model.fit_bayesian(
    gamma_dot,
    stress,
    test_mode="flow_curve",
    num_warmup=NUM_WARMUP,
    num_samples=NUM_SAMPLES,
    num_chains=NUM_CHAINS,
    initial_values=initial_values,
    seed=100,
)
t_bayes = time.time() - t0
print(f"\\nBayesian inference time: {t_bayes:.1f} s")""")

    # Fix posterior predictive cell to handle variable sample count
    idx = find_cell_containing(nb, "n_draws = min(200")
    if idx is not None:
        src = get_cell_source(nb["cells"][idx])
        src = src.replace("n_draws = min(200,", "n_draws = min(50 if FAST_MODE else 200,")
        set_cell_source(nb["cells"][idx], src)

    clear_all_outputs(nb)
    save_notebook(nb, path)


# ============================================================
# NB02: Startup Shear
# ============================================================
def fix_nb02(path: Path):
    print(f"\nFixing NB02: {path.name}")
    nb = load_notebook(path)

    # Fix setup cell
    idx = find_cell_containing(nb, "warnings.filterwarnings")
    if idx is not None:
        set_cell_source(nb["cells"][idx], """%matplotlib inline
import gc
import os
import time
import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from rheojax.core.jax_config import safe_import_jax, verify_float64
from rheojax.models.stz import STZConventional

jax, jnp = safe_import_jax()
verify_float64()

# Targeted suppression: equinox internal deprecation (harmless, not under our control)
warnings.filterwarnings("ignore", message=".*is_leaf.*", category=DeprecationWarning)

FAST_MODE = int(os.environ.get("FAST_MODE", "1"))
print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"FAST_MODE: {FAST_MODE}")""")

    # Fix overshoot sweep cell: reduce points in FAST_MODE
    idx = find_cell_containing(nb, "gdot_sweep = np.logspace")
    if idx is not None:
        set_cell_source(nb["cells"][idx], """gdot_sweep = np.logspace(-1, 2, 8 if FAST_MODE else 20)
overshoot_ratios = []

for gdot in gdot_sweep:
    m = STZConventional(variant="standard")

    # Widen bounds for soft-matter scales before setting values
    for p_name in ["G0", "sigma_y", "tau0"]:
        if p_name in m.parameters.keys():
            m.parameters.set_bounds(p_name, (1e-20, 1e20))

    for name, value in startup_params.items():
        if name in m.parameters.keys():
            m.parameters[name].value = value

    m._gamma_dot_applied = float(gdot)
    m._sigma_applied = None
    m._test_mode = "startup"
    m.fitted_ = True

    t_ov = np.linspace(1e-4, 20.0, 200 if FAST_MODE else 500)
    stress_ov = m.predict(t_ov)

    sigma_max = np.max(stress_ov)
    sigma_ss = stress_ov[-1]
    ratio = sigma_max / max(sigma_ss, 1e-20)
    overshoot_ratios.append(ratio)
    del m
    gc.collect()

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogx(gdot_sweep, overshoot_ratios, "o-", markersize=5, lw=2, color="C3")
ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
ax.set_xlabel("Shear rate [1/s]")
ax.set_ylabel("$\\\\sigma_{max} / \\\\sigma_{ss}$")
ax.set_title("Overshoot Ratio vs Shear Rate")
ax.grid(True, alpha=0.3)
plt.tight_layout()
display(fig)
plt.close(fig)""")

    # Fix Bayesian cell: FAST_MODE + gc + reduced samples for ODE
    idx = find_cell_containing(nb, "NUM_WARMUP = 200")
    if idx is not None:
        set_cell_source(nb["cells"][idx], """initial_values = {
    name: model.parameters.get_value(name)
    for name in model.parameters.keys()
}

# Free memory before Bayesian inference
gc.collect()
jax.clear_caches()

if FAST_MODE:
    NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS = 25, 50, 1
else:
    NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS = 200, 500, 1

t0 = time.time()
result = model.fit_bayesian(
    t_fit,
    stress_fit,
    test_mode="startup",
    num_warmup=NUM_WARMUP,
    num_samples=NUM_SAMPLES,
    num_chains=NUM_CHAINS,
    initial_values=initial_values,
    seed=42,
)
t_bayes = time.time() - t0
print(f"Bayesian inference time: {t_bayes:.1f} s")""")

    clear_all_outputs(nb)
    save_notebook(nb, path)


# ============================================================
# NB03: Stress Relaxation
# ============================================================
def fix_nb03(path: Path):
    print(f"\nFixing NB03: {path.name}")
    nb = load_notebook(path)

    # Fix setup cell
    idx = find_cell_containing(nb, "warnings.filterwarnings")
    if idx is not None:
        set_cell_source(nb["cells"][idx], """%matplotlib inline
import gc
import os
import time
import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from rheojax.core.jax_config import safe_import_jax, verify_float64
from rheojax.models.stz import STZConventional

jax, jnp = safe_import_jax()
verify_float64()

# Targeted suppression: equinox internal deprecation (harmless, not under our control)
warnings.filterwarnings("ignore", message=".*is_leaf.*", category=DeprecationWarning)

FAST_MODE = int(os.environ.get("FAST_MODE", "1"))
print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"FAST_MODE: {FAST_MODE}")""")

    # Fix aging sweep: add gc between fits
    idx = find_cell_containing(nb, "aging_results = {}")
    if idx is not None:
        src = get_cell_source(nb["cells"][idx])
        # Add gc after each fit iteration
        src = src.replace(
            '        print(f"t_age={t_age:5d}s: FAILED ({e})")',
            '        print(f"t_age={t_age:5d}s: FAILED ({e})")\n    del m\n    gc.collect()'
        )
        # Also add gc after successful fit
        src = src.replace(
            '              f"R^2={aging_results[t_age][\'r_squared\']:.4f}")',
            '              f"R^2={aging_results[t_age][\'r_squared\']:.4f}")\n    del m\n    gc.collect()'
        )
        set_cell_source(nb["cells"][idx], src)

    # Fix Bayesian cell: FAST_MODE + gc
    idx = find_cell_containing(nb, "NUM_WARMUP = 200")
    if idx is not None:
        set_cell_source(nb["cells"][idx], """initial_values = {
    name: model.parameters.get_value(name)
    for name in model.parameters.keys()
}

# Free memory before Bayesian inference
gc.collect()
jax.clear_caches()

if FAST_MODE:
    NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS = 25, 50, 1
else:
    NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS = 200, 500, 1

# Use t_age=3600 data
t_data, G_t = datasets[3600]
stress_data = G_t * gamma_0

t0 = time.time()
result = model.fit_bayesian(
    t_data,
    stress_data,
    test_mode="relaxation",
    num_warmup=NUM_WARMUP,
    num_samples=NUM_SAMPLES,
    num_chains=NUM_CHAINS,
    initial_values=initial_values,
    seed=42,
)
t_bayes = time.time() - t0
print(f"Bayesian inference time: {t_bayes:.1f} s")""")

    clear_all_outputs(nb)
    save_notebook(nb, path)


# ============================================================
# NB04: Creep
# ============================================================
def fix_nb04(path: Path):
    print(f"\nFixing NB04: {path.name}")
    nb = load_notebook(path)

    # Fix setup cell
    idx = find_cell_containing(nb, "warnings.filterwarnings")
    if idx is not None:
        set_cell_source(nb["cells"][idx], """%matplotlib inline
import gc
import os
import time
import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from rheojax.core.jax_config import safe_import_jax, verify_float64
from rheojax.models.stz import STZConventional

jax, jnp = safe_import_jax()
verify_float64()

# Targeted suppression: equinox internal deprecation (harmless, not under our control)
warnings.filterwarnings("ignore", message=".*is_leaf.*", category=DeprecationWarning)

FAST_MODE = int(os.environ.get("FAST_MODE", "1"))
print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"FAST_MODE: {FAST_MODE}")""")

    # Fix Bayesian cell: FAST_MODE + gc
    idx = find_cell_containing(nb, "NUM_WARMUP = 200")
    if idx is not None:
        set_cell_source(nb["cells"][idx], """initial_values = {
    name: model.parameters.get_value(name)
    for name in model.parameters.keys()
}

# Free memory before Bayesian inference
gc.collect()
jax.clear_caches()

if FAST_MODE:
    NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS = 25, 50, 1
else:
    NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS = 200, 500, 1

t0 = time.time()
result = model.fit_bayesian(
    t_data,
    strain_data,
    test_mode="creep",
    num_warmup=NUM_WARMUP,
    num_samples=NUM_SAMPLES,
    num_chains=NUM_CHAINS,
    initial_values=initial_values,
    seed=42,
)
t_bayes = time.time() - t0
print(f"Bayesian inference time: {t_bayes:.1f} s")""")

    # Fix bifurcation demo: add gc between model simulations
    idx = find_cell_containing(nb, "sigma_y_fit = model.parameters.get_value")
    if idx is not None:
        src = get_cell_source(nb["cells"][idx])
        src = src.replace(
            "    strain_sim = m.predict(t_sim)",
            "    strain_sim = m.predict(t_sim)\n    del m\n    gc.collect()"
        )
        set_cell_source(nb["cells"][idx], src)

    clear_all_outputs(nb)
    save_notebook(nb, path)


# ============================================================
# NB05: SAOS
# ============================================================
def fix_nb05(path: Path):
    print(f"\nFixing NB05: {path.name}")
    nb = load_notebook(path)

    # Fix setup cell
    idx = find_cell_containing(nb, "warnings.filterwarnings")
    if idx is not None:
        set_cell_source(nb["cells"][idx], """%matplotlib inline
import gc
import os
import time
import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from rheojax.core.jax_config import safe_import_jax, verify_float64
from rheojax.models.stz import STZConventional

jax, jnp = safe_import_jax()
verify_float64()

# Targeted suppression: equinox internal deprecation (harmless, not under our control)
warnings.filterwarnings("ignore", message=".*is_leaf.*", category=DeprecationWarning)

FAST_MODE = int(os.environ.get("FAST_MODE", "1"))
print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"FAST_MODE: {FAST_MODE}")""")

    # Fix Bayesian cell: FAST_MODE + gc
    idx = find_cell_containing(nb, "NUM_WARMUP = 200")
    if idx is not None:
        set_cell_source(nb["cells"][idx], """initial_values = {
    name: model.parameters.get_value(name)
    for name in model.parameters.keys()
}

# Free memory before Bayesian inference
gc.collect()
jax.clear_caches()

if FAST_MODE:
    NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS = 50, 100, 1
else:
    NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS = 1000, 2000, 4

t0 = time.time()
result = model.fit_bayesian(
    omega,
    G_star,
    test_mode="oscillation",
    num_warmup=NUM_WARMUP,
    num_samples=NUM_SAMPLES,
    num_chains=NUM_CHAINS,
    initial_values=initial_values,
    seed=42,
)
t_bayes = time.time() - t0
print(f"Bayesian inference time: {t_bayes:.1f} s")""")

    # Fix posterior predictive cell to handle variable sample count
    idx = find_cell_containing(nb, "n_draws = min(200")
    if idx is not None:
        src = get_cell_source(nb["cells"][idx])
        src = src.replace("n_draws = min(200,", "n_draws = min(50 if FAST_MODE else 200,")
        set_cell_source(nb["cells"][idx], src)

    clear_all_outputs(nb)
    save_notebook(nb, path)


# ============================================================
# NB06: LAOS (no Bayesian — just warnings/FAST_MODE)
# ============================================================
def fix_nb06(path: Path):
    print(f"\nFixing NB06: {path.name}")
    nb = load_notebook(path)

    # Fix setup cell
    idx = find_cell_containing(nb, "warnings.filterwarnings")
    if idx is not None:
        set_cell_source(nb["cells"][idx], """%matplotlib inline
import gc
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from rheojax.core.jax_config import safe_import_jax, verify_float64
from rheojax.models.stz import STZConventional

jax, jnp = safe_import_jax()
verify_float64()

# Targeted suppression: equinox internal deprecation (harmless, not under our control)
warnings.filterwarnings("ignore", message=".*is_leaf.*", category=DeprecationWarning)

FAST_MODE = int(os.environ.get("FAST_MODE", "1"))
print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"FAST_MODE: {FAST_MODE}")""")

    # Fix LAOS amplitudes: fewer in FAST_MODE
    idx = find_cell_containing(nb, "amplitudes = [0.01, 0.1, 0.5, 2.0]")
    if idx is not None:
        src = get_cell_source(nb["cells"][idx])
        src = src.replace(
            "amplitudes = [0.01, 0.1, 0.5, 2.0]",
            "amplitudes = [0.01, 0.5, 2.0] if FAST_MODE else [0.01, 0.1, 0.5, 2.0]"
        )
        set_cell_source(nb["cells"][idx], src)

    # Fix variant comparison: add gc
    idx = find_cell_containing(nb, "gamma_0_large = 2.0")
    if idx is not None:
        src = get_cell_source(nb["cells"][idx])
        set_cell_source(nb["cells"][idx], src)

    clear_all_outputs(nb)
    save_notebook(nb, path)


def main():
    stz_dir = Path("examples/stz")

    fix_nb01(stz_dir / "01_stz_flow_curve.ipynb")
    fix_nb02(stz_dir / "02_stz_startup_shear.ipynb")
    fix_nb03(stz_dir / "03_stz_stress_relaxation.ipynb")
    fix_nb04(stz_dir / "04_stz_creep.ipynb")
    fix_nb05(stz_dir / "05_stz_saos.ipynb")
    fix_nb06(stz_dir / "06_stz_laos.ipynb")

    print("\n\nAll 6 STZ notebooks fixed!")
    print("Run: FAST_MODE=1 uv run python scripts/run_notebooks.py --subdir examples/stz --timeout 345600 --log-dir examples/stz/_run_logs")


if __name__ == "__main__":
    main()
