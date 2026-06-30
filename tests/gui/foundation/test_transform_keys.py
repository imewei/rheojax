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
