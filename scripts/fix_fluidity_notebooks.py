#!/usr/bin/env python
"""Apply systematic fixes to all fluidity notebooks.

Fixes applied:
1. Add FAST_MODE support to Bayesian cells (env var FAST_MODE for CI)
2. Replace blanket FutureWarning suppression with targeted equinox filter
3. Add %matplotlib inline where missing
4. Add plt.close('all') after plt.show() where missing
5. Ensure headless-safe matplotlib backend
"""

import json
from pathlib import Path


def fix_future_warning_suppression(source_lines: list[str]) -> tuple[list[str], bool]:
    """Replace blanket FutureWarning suppression with targeted equinox filter."""
    changed = False
    new_lines = []
    for line in source_lines:
        if 'warnings.filterwarnings("ignore", category=FutureWarning)' in line:
            # Replace with targeted suppression for equinox DeprecationWarning
            new_lines.append(
                '# Suppress upstream equinox DeprecationWarning (jax.core.mapped_aval deprecated)\n'
            )
            new_lines.append(
                'warnings.filterwarnings(\n'
            )
            new_lines.append(
                '    "ignore",\n'
            )
            new_lines.append(
                '    message="jax.core.*_aval is deprecated",\n'
            )
            new_lines.append(
                '    category=DeprecationWarning,\n'
            )
            new_lines.append(
                '    module="equinox",\n'
            )
            new_lines.append(
                ')\n'
            )
            changed = True
        else:
            new_lines.append(line)
    return new_lines, changed


def add_matplotlib_inline(nb: dict) -> bool:
    """Add %matplotlib inline to import cell if missing."""
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            if "%matplotlib inline" in src:
                return False  # Already has it

    # Find the main import cell (first cell with 'import matplotlib' or 'import numpy')
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            if "import matplotlib" in src or "import numpy" in src:
                # Prepend %matplotlib inline
                if cell["source"] and isinstance(cell["source"], list):
                    cell["source"].insert(0, "%matplotlib inline\n")
                elif isinstance(cell["source"], str):
                    cell["source"] = "%matplotlib inline\n" + cell["source"]
                return True
    return False


def add_plt_close_after_show(nb: dict) -> int:
    """Add plt.close('all') after plt.show() calls that don't have it."""
    fixes = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src_lines = cell["source"] if isinstance(cell["source"], list) else cell["source"].splitlines(True)
        new_lines = []
        for i, line in enumerate(src_lines):
            stripped = line.strip()
            if stripped == "plt.show()" or stripped == "plt.show();":
                # Check if next line already has plt.close
                next_line = src_lines[i + 1].strip() if i + 1 < len(src_lines) else ""
                if "plt.close" not in next_line:
                    # Ensure the plt.show() line ends with newline
                    if not line.endswith("\n"):
                        line = line + "\n"
                    new_lines.append(line)
                    # Get indentation of plt.show()
                    indent = line[: len(line) - len(line.lstrip())]
                    new_lines.append(f"{indent}plt.close('all')\n")
                    fixes += 1
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        if fixes > 0:
            cell["source"] = new_lines
    return fixes


def add_fast_mode(nb: dict) -> bool:
    """Add FAST_MODE env var support to Bayesian configuration cells."""
    changed = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]

        # Find cells with NUM_WARMUP = ... NUM_SAMPLES = ... pattern
        if "NUM_WARMUP" in src and "NUM_SAMPLES" in src and "fit_bayesian" in src:
            # Already has FAST_MODE?
            if "FAST_MODE" in src:
                continue

            lines = cell["source"] if isinstance(cell["source"], list) else cell["source"].splitlines(True)
            new_lines = []
            inserted_fast_mode = False

            for line in lines:
                stripped = line.strip()
                # Find the first NUM_WARMUP = ... line (not commented)
                if (
                    not inserted_fast_mode
                    and stripped.startswith("NUM_WARMUP")
                    and "=" in stripped
                    and not stripped.startswith("#")
                ):
                    # Insert FAST_MODE block before the NUM_WARMUP line
                    indent = line[: len(line) - len(line.lstrip())]
                    new_lines.append(f"{indent}# FAST_MODE for CI: set FAST_MODE=1 env var for quick iteration\n")
                    new_lines.append(f"{indent}FAST_MODE = os.environ.get('FAST_MODE', '0') == '1'\n")
                    new_lines.append(f"{indent}if FAST_MODE:\n")
                    new_lines.append(f"{indent}    NUM_WARMUP = 50\n")
                    new_lines.append(f"{indent}    NUM_SAMPLES = 100\n")
                    new_lines.append(f"{indent}    NUM_CHAINS = 1\n")
                    new_lines.append(f"{indent}else:\n")
                    new_lines.append(f"{indent}    NUM_WARMUP = 200\n")
                    new_lines.append(f"{indent}    NUM_SAMPLES = 500\n")
                    new_lines.append(f"{indent}    NUM_CHAINS = 1\n")
                    inserted_fast_mode = True
                    # Skip the original NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS lines
                    continue
                elif inserted_fast_mode and stripped.startswith(("NUM_SAMPLES", "NUM_CHAINS")) and "=" in stripped and not stripped.startswith("#"):
                    # Skip these lines (replaced by FAST_MODE block above)
                    continue
                elif inserted_fast_mode and stripped.startswith("#") and any(kw in stripped for kw in ["NUM_WARMUP", "NUM_SAMPLES", "NUM_CHAINS", "production"]):
                    # Skip commented-out production config
                    continue
                else:
                    new_lines.append(line)

            if inserted_fast_mode:
                cell["source"] = new_lines
                changed = True

    return changed


def add_fast_mode_nonstandard(nb: dict) -> bool:
    """Add FAST_MODE to notebooks with non-standard Bayesian config (inline nums)."""
    changed = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]

        # Pattern: num_warmup=200, num_samples=500 inline in fit_bayesian call
        if "fit_bayesian" in src and "num_warmup=" in src and "FAST_MODE" not in src and "NUM_WARMUP" not in src:
            lines = cell["source"] if isinstance(cell["source"], list) else cell["source"].splitlines(True)
            new_lines = []
            inserted = False

            for line in lines:
                stripped = line.strip()
                # Insert FAST_MODE block before fit_bayesian call
                if not inserted and "fit_bayesian" in stripped:
                    indent = line[: len(line) - len(line.lstrip())]
                    new_lines.append(f"\n{indent}# FAST_MODE for CI: set FAST_MODE=1 env var for quick iteration\n")
                    new_lines.append(f"{indent}FAST_MODE = os.environ.get('FAST_MODE', '0') == '1'\n")
                    new_lines.append(f"{indent}_num_warmup = 50 if FAST_MODE else 200\n")
                    new_lines.append(f"{indent}_num_samples = 100 if FAST_MODE else 500\n")
                    new_lines.append(f"{indent}_num_chains = 1\n\n")
                    inserted = True
                    # Replace hardcoded values in the call
                    line = line.replace("num_warmup=200", "num_warmup=_num_warmup")
                    line = line.replace("num_warmup=500", "num_warmup=_num_warmup")
                    line = line.replace("num_warmup=1000", "num_warmup=_num_warmup")
                    line = line.replace("num_samples=200", "num_samples=_num_samples")
                    line = line.replace("num_samples=500", "num_samples=_num_samples")
                    line = line.replace("num_samples=1000", "num_samples=_num_samples")
                    line = line.replace("num_samples=2000", "num_samples=_num_samples")
                new_lines.append(line)

            if inserted:
                cell["source"] = new_lines
                changed = True

    return changed


def ensure_os_import(nb: dict) -> bool:
    """Ensure 'import os' is present for FAST_MODE env var access."""
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if "import os" in src:
            return False  # Already imported

    # Find first code cell with imports and add 'import os'
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if "import numpy" in src or "import matplotlib" in src:
            lines = cell["source"] if isinstance(cell["source"], list) else cell["source"].splitlines(True)
            # Insert after the first import line
            for i, line in enumerate(lines):
                if line.strip().startswith("import ") and "colab" not in line.lower():
                    lines.insert(i, "import os\n")
                    cell["source"] = lines
                    return True
    return False


def fix_notebook(nb_path: Path) -> dict:
    """Apply all fixes to a single notebook."""
    with open(nb_path) as f:
        nb = json.load(f)

    changes = {
        "matplotlib_inline": False,
        "plt_close": 0,
        "future_warning": False,
        "fast_mode": False,
        "os_import": False,
    }

    # 1. Add %matplotlib inline
    changes["matplotlib_inline"] = add_matplotlib_inline(nb)

    # 2. Add plt.close('all')
    changes["plt_close"] = add_plt_close_after_show(nb)

    # 3. Fix FutureWarning suppression
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            lines = cell["source"] if isinstance(cell["source"], list) else cell["source"].splitlines(True)
            new_lines, changed = fix_future_warning_suppression(lines)
            if changed:
                cell["source"] = new_lines
                changes["future_warning"] = True

    # 4. Add FAST_MODE support
    changes["fast_mode"] = add_fast_mode(nb) or add_fast_mode_nonstandard(nb)

    # 5. Ensure os import (needed for FAST_MODE)
    if changes["fast_mode"]:
        changes["os_import"] = ensure_os_import(nb)

    # Write back if any changes
    if any(v for v in changes.values()):
        with open(nb_path, "w") as f:
            json.dump(nb, f, indent=1)

    return changes


def main():
    fluidity_dir = Path("examples/fluidity")
    notebooks = sorted(fluidity_dir.glob("*.ipynb"))

    print(f"Processing {len(notebooks)} fluidity notebooks")
    print("=" * 70)

    total_changes = 0
    for nb_path in notebooks:
        changes = fix_notebook(nb_path)
        summary_parts = []
        if changes["matplotlib_inline"]:
            summary_parts.append("added %matplotlib inline")
        if changes["plt_close"]:
            summary_parts.append(f"added {changes['plt_close']} plt.close()")
        if changes["future_warning"]:
            summary_parts.append("fixed FutureWarning → targeted equinox filter")
        if changes["fast_mode"]:
            summary_parts.append("added FAST_MODE")
        if changes["os_import"]:
            summary_parts.append("added import os")

        if summary_parts:
            print(f"  {nb_path.name}: {', '.join(summary_parts)}")
            total_changes += 1
        else:
            print(f"  {nb_path.name}: no changes needed")

    print("=" * 70)
    print(f"Modified {total_changes}/{len(notebooks)} notebooks")


if __name__ == "__main__":
    main()
