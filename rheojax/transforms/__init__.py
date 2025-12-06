"""Data analysis transforms for rheological data.

This package contains 7 data transforms for rheological analysis:

Transforms (7):
    - FFTAnalysis: FFT-based frequency spectrum analysis
    - Mastercurve: Time-temperature superposition (TTS) mastercurves
    - SRFS: Strain-rate frequency superposition (analogous to TTS)
    - MutationNumber: Viscoelastic character analysis (0=elastic, 1=viscous)
    - OWChirp: OWChirp transform for LAOS analysis
    - SmoothDerivative: Smooth noise-robust differentiation
    - SPPDecomposer: Sequence of Physical Processes (SPP) for LAOS analysis
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
