"""Headless regression for transform presets and data loading."""

import numpy as np

from rheojax.gui.services.data_service import DataService
from rheojax.gui.services.transform_service import TransformService


def test_fft_transform_on_synthetic_dataset(tmp_path) -> None:
    """Synthetic CSV load + FFT transform sanity."""

    # Create small synthetic dataset emulating notebook frequency sweep
    x = np.linspace(0, 10, 200)
    y = np.sin(2 * np.pi * 1.0 * x) + 0.1 * np.random.randn(x.size)
    csv_path = tmp_path / "synthetic_freq.csv"
    np.savetxt(csv_path, np.column_stack([x, y]), delimiter=",", header="time,signal", comments="")

    data_service = DataService()
    transform_service = TransformService()

    rheo_data = data_service.load_file(str(csv_path), x_col="time", y_col="signal")
    result = transform_service.apply_transform(
        "fft",
        rheo_data,
        {
            "direction": "forward",
            "window": "hann",
            "detrend": True,
            "normalize": True,
            "return_psd": False,
        },
    )

    # Basic sanity: output has numeric arrays and non-empty content
    assert hasattr(result, "x") and hasattr(result, "y")
    # Accept numpy or JAX arrays
    assert hasattr(result.x, "shape") and hasattr(result.y, "shape")
    assert result.x.shape[0] == result.y.shape[0]
    assert result.x.shape[0] > 50
