"""Tests for batch transform vectorization."""

import numpy as np
import pytest


class TestBatchFFT:
    @pytest.mark.smoke
    def test_batch_fft_same_shape(self):
        from rheojax.core.data import RheoData
        from rheojax.transforms.fft_analysis import FFTAnalysis

        fft = FFTAnalysis()
        t = np.linspace(0, 10, 256)
        datasets = [
            RheoData(x=t, y=np.sin(2 * np.pi * f * t)) for f in [1.0, 2.0, 3.0]
        ]
        results = fft.batch_transform(datasets)
        assert len(results) == 3
        for r in results:
            assert r.x is not None
            assert r.y is not None

    def test_batch_fft_matches_sequential(self):
        from rheojax.core.data import RheoData
        from rheojax.transforms.fft_analysis import FFTAnalysis

        fft = FFTAnalysis()
        t = np.linspace(0, 10, 256)
        datasets = [
            RheoData(x=t, y=np.sin(2 * np.pi * f * t)) for f in [1.0, 2.0]
        ]

        sequential = [fft.transform(d) for d in datasets]
        batched = fft.batch_transform(datasets)

        for s, b in zip(sequential, batched):
            np.testing.assert_allclose(
                np.asarray(s.y), np.asarray(b.y), rtol=1e-10
            )

    def test_batch_fft_variable_length_fallback(self):
        from rheojax.core.data import RheoData
        from rheojax.transforms.fft_analysis import FFTAnalysis

        fft = FFTAnalysis()
        datasets = [
            RheoData(
                x=np.linspace(0, 10, 128), y=np.sin(np.linspace(0, 10, 128))
            ),
            RheoData(
                x=np.linspace(0, 10, 256), y=np.sin(np.linspace(0, 10, 256))
            ),
        ]
        # Should still work (falls back to sequential)
        results = fft.batch_transform(datasets)
        assert len(results) == 2

    def test_batch_fft_empty(self):
        from rheojax.transforms.fft_analysis import FFTAnalysis

        fft = FFTAnalysis()
        results = fft.batch_transform([])
        assert results == []


class TestBatchSmoothDerivative:
    @pytest.mark.smoke
    def test_batch_smooth_derivative_same_shape(self):
        from rheojax.core.data import RheoData
        from rheojax.transforms.smooth_derivative import SmoothDerivative

        sd = SmoothDerivative()
        t = np.linspace(0, 10, 100)
        datasets = [RheoData(x=t, y=np.sin(f * t)) for f in [1.0, 2.0, 3.0]]
        results = sd.batch_transform(datasets)
        assert len(results) == 3

    def test_batch_smooth_derivative_matches_sequential(self):
        from rheojax.core.data import RheoData
        from rheojax.transforms.smooth_derivative import SmoothDerivative

        sd = SmoothDerivative()
        t = np.linspace(0, 10, 100)
        datasets = [RheoData(x=t, y=np.sin(f * t)) for f in [1.0, 2.0]]

        sequential = [sd.transform(d) for d in datasets]
        batched = sd.batch_transform(datasets)

        for s, b in zip(sequential, batched):
            np.testing.assert_allclose(
                np.asarray(s.y), np.asarray(b.y), rtol=1e-10
            )
