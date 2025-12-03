"""Data analysis transforms for rheological data.

This module provides JAX-accelerated implementations of:
- FFT-based frequency analysis
- Time-temperature superposition (mastercurves)
- Strain-rate frequency superposition (SRFS)
- Mutation number analysis
- OWChirp transform for LAOS analysis
- Smooth noise-robust differentiation
"""

from rheojax.transforms.fft_analysis import FFTAnalysis
from rheojax.transforms.mastercurve import Mastercurve
from rheojax.transforms.mutation_number import MutationNumber
from rheojax.transforms.owchirp import OWChirp
from rheojax.transforms.smooth_derivative import SmoothDerivative
from rheojax.transforms.spp_decomposer import SPPDecomposer, spp_analyze
from rheojax.transforms.srfs import (
    SRFS,
    compute_shear_band_coexistence,
    detect_shear_banding,
)

__all__ = [
    "FFTAnalysis",
    "Mastercurve",
    "MutationNumber",
    "OWChirp",
    "SmoothDerivative",
    "SPPDecomposer",
    "spp_analyze",
    "SRFS",
    "detect_shear_banding",
    "compute_shear_band_coexistence",
]
