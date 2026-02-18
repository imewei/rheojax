import argparse
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient

# 96 hours in seconds
TIMEOUT_96H = 96 * 60 * 60


def setup_environment() -> dict[str, str]:
    """Set up deterministic environment variables."""
    print("Setting up environment...")
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["JAX_ENABLE_X64"] = "True"
    env["PYTHONHASHSEED"] = "42"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
    env["FAST_MODE"] = "0"  # Ensure slow mode for deep testing

    # Isolate Jupyter environment to avoid PermissionError/Config issues
    import tempfile

    temp_jupyter_dir = tempfile.mkdtemp(prefix="jupyter_runtime_")
    env["JUPYTER_CONFIG_DIR"] = os.path.join(temp_jupyter_dir, "config")
    env["JUPYTER_DATA_DIR"] = os.path.join(temp_jupyter_dir, "data")
    env["JUPYTER_RUNTIME_DIR"] = os.path.join(temp_jupyter_dir, "runtime")
    env["IPYTHONDIR"] = os.path.join(temp_jupyter_dir, "ipython")

    return env


def main():
    parser = argparse.ArgumentParser(
        description="Run a SINGLE notebook with 96h timeout"
    )
    parser.add_argument("notebook_path", type=Path, help="Path to the notebook file")
    args = parser.parse_args()

    notebook_path = args.notebook_path.absolute()
    if not notebook_path.exists():
        print(f"Error: {notebook_path} does not exist")
        sys.exit(1)

    print(f"Running notebook: {notebook_path}")

    # Set environment variables
    env_vars = setup_environment()
    os.environ.update(env_vars)

    start_time = datetime.now()
    _log_dir = notebook_path.parent

    try:
        nb = nbformat.read(notebook_path, as_version=4)
        original_cwd = os.getcwd()
        notebook_dir = notebook_path.parent
        os.chdir(notebook_dir)

        client = NotebookClient(
            nb,
            timeout=TIMEOUT_96H,
            kernel_name="python3",
            allow_errors=False,
            resources={"metadata": {"path": str(notebook_dir)}},
        )

        print("Executing...")
        client.execute()

        # Save executed notebook
        nbformat.write(nb, notebook_path.name)
        print("SUCCESS: Notebook executed and saved.")

    except Exception as e:
        print(f"FAILURE: {type(e).__name__}: {e}")
        traceback.print_exc()
        # Save partial notebook for debugging
        try:
            nbformat.write(nb, notebook_path.name)
            print(f"Saved partial notebook to {notebook_path.name}")
        except Exception:
            print("Could not save partial notebook")
        sys.exit(1)
    finally:
        os.chdir(original_cwd)
        os.chdir(original_cwd)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"Runtime: {duration:.1f} seconds")


if __name__ == "__main__":
    main()
