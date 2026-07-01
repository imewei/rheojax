import nlsq  # noqa: F401 — must precede rheojax.core imports (float64 init)

from rheojax.gui.services.transform_service import TransformService


def test_registry_keys_accepted():
    svc = TransformService()
    keys = set(svc.get_available_transforms())  # existing method
    assert {"fft_analysis", "smooth_derivative", "spp_decomposer"} <= keys
    # legacy aliases still resolve to the same class via the new resolve() wrapper
    assert svc.resolve("fft") is svc.resolve("fft_analysis")


def test_resolve_returns_class():
    svc = TransformService()
    from rheojax.transforms.fft_analysis import FFTAnalysis
    from rheojax.transforms.smooth_derivative import SmoothDerivative
    from rheojax.transforms.spp_decomposer import SPPDecomposer

    assert svc.resolve("fft") is FFTAnalysis
    assert svc.resolve("fft_analysis") is FFTAnalysis
    assert svc.resolve("derivative") is SmoothDerivative
    assert svc.resolve("smooth_derivative") is SmoothDerivative
    assert svc.resolve("spp") is SPPDecomposer
    assert svc.resolve("spp_decomposer") is SPPDecomposer


def test_legacy_keys_still_in_available():
    svc = TransformService()
    keys = set(svc.get_available_transforms())
    assert {"fft", "derivative", "spp"} <= keys


def test_apply_transform_unified_key_fft():
    """apply_transform accepts unified key 'fft_analysis' without raising."""
    import numpy as np
    from rheojax.core.data import RheoData

    svc = TransformService()
    t = np.linspace(0, 1, 128)
    data = RheoData(x=t, y=np.sin(2 * np.pi * 5 * t), initial_test_mode="oscillation")
    result = svc.apply_transform("fft_analysis", data, {})
    assert result is not None


def test_apply_transform_unified_key_derivative():
    """apply_transform accepts unified key 'smooth_derivative' without raising."""
    import numpy as np
    from rheojax.core.data import RheoData

    svc = TransformService()
    x = np.linspace(0.1, 10.0, 50)
    data = RheoData(x=x, y=x**2, initial_test_mode="relaxation")
    result = svc.apply_transform("smooth_derivative", data, {})
    assert result is not None
