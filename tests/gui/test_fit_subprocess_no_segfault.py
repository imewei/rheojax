import os
import subprocess
import sys

import pytest


@pytest.mark.gui
def test_fit_subprocess_exits_cleanly() -> None:
    """Run a minimal offscreen fit in a subprocess.

    This catches hard crashes (e.g. segfaults) that pytest cannot intercept.
    """

    script = r"""
import os
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication


def main() -> int:
    from rheojax.gui.app.main_window import RheoJAXMainWindow
    import numpy as np

    app = QApplication.instance() or QApplication([])
    w = RheoJAXMainWindow()
    w.show()

    def shutdown() -> None:
        try:
            if getattr(w, "worker_pool", None) is not None:
                w.worker_pool.shutdown(wait=True, timeout_ms=5000)
        except Exception:
            pass
        app.exit(0)

    def on_job_completed(job_id: str, _result: object) -> None:
        QTimer.singleShot(0, shutdown)

    def on_job_failed(job_id: str, _error: str) -> None:
        QTimer.singleShot(0, shutdown)

    w.worker_pool.job_completed.connect(on_job_completed)
    w.worker_pool.job_failed.connect(on_job_failed)

    def start_flow() -> None:
        x = np.logspace(-2, 2, 200)
        y = 1e3 * np.exp(-x / 0.5) + 10.0
        dataset_id = "subprocess"
        w.store.dispatch(
            "IMPORT_DATA_SUCCESS",
            {
                "dataset_id": dataset_id,
                "name": "Subprocess Dataset",
                "test_mode": "relaxation",
                "file_path": None,
                "x_data": x,
                "y_data": y,
                "y2_data": None,
                "metadata": {"test_mode": "relaxation", "domain": "time"},
            },
        )
        w.store.dispatch("SET_ACTIVE_DATASET", {"dataset_id": dataset_id})
        model = "maxwell"
        w.store.dispatch("SET_ACTIVE_MODEL", {"model_name": model})
        w.navigate_to("fit")
        w._on_fit_requested_from_page({"model_name": model, "dataset_id": dataset_id})

    QTimer.singleShot(0, start_flow)
    QTimer.singleShot(20_000, shutdown)
    return int(app.exec())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
"""

    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")

    completed = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        timeout=30,
    )
    assert completed.returncode == 0
