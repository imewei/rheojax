"""Data analysis transforms for rheological data.

This package contains 11 data transforms for rheological analysis:

Transforms (11):
    - FFTAnalysis: FFT-based frequency spectrum analysis
    - Mastercurve: Time-temperature superposition (TTS) mastercurves
    - SRFS: Strain-rate frequency superposition (analogous to TTS)
    - MutationNumber: Viscoelastic character analysis (0=elastic, 1=viscous)
    - OWChirp: OWChirp transform for LAOS analysis
    - SmoothDerivative: Smooth noise-robust differentiation
    - SPPDecomposer: Sequence of Physical Processes (SPP) for LAOS analysis
    - PronyConversion: Time ↔ frequency domain via Prony series
    - CoxMerz: Cox-Merz rule validation (|η*| vs η)
    - LVEEnvelope: Linear viscoelastic startup stress envelope
    - SpectrumInversion: Relaxation spectrum H(τ) recovery
"""

from rheojax.transforms.cox_merz import CoxMerz
from rheojax.transforms.fft_analysis import FFTAnalysis
from rheojax.transforms.lve_envelope import LVEEnvelope
from rheojax.transforms.mastercurve import Mastercurve
from rheojax.transforms.mutation_number import MutationNumber
from rheojax.transforms.owchirp import OWChirp
from rheojax.transforms.prony_conversion import PronyConversion
from rheojax.transforms.smooth_derivative import SmoothDerivative
from rheojax.transforms.spectrum_inversion import SpectrumInversion
from rheojax.transforms.spp_decomposer import SPPDecomposer, spp_analyze
from rheojax.transforms.srfs import (
    SRFS,
    compute_shear_band_coexistence,
    detect_shear_banding,
)


def _ensure_all_registered() -> None:
    """No-op — all transforms are eagerly imported above.

    This function exists so that ``PipelineBuilder._validate_components()``
    can call ``from rheojax.transforms import _ensure_all_registered``
    symmetrically with the models package. Importing this module already
    triggers all ``@TransformRegistry.register`` decorators.
    """


__all__ = [
    "CoxMerz",
    "FFTAnalysis",
    "LVEEnvelope",
    "Mastercurve",
    "MutationNumber",
    "OWChirp",
    "PronyConversion",
    "SmoothDerivative",
    "SpectrumInversion",
    "SPPDecomposer",
    "spp_analyze",
    "SRFS",
    "detect_shear_banding",
    "compute_shear_band_coexistence",
]
