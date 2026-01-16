"""Core definitions for the Protocol-Driven Inventory System.

This module defines the classifications used to categorize models and transforms
in the RheoJAX registry. It provides the type system for:
1. Models (via Protocol)
2. Transforms (via TransformType)
"""

from __future__ import annotations

from enum import Enum


class Protocol(str, Enum):
    """Rheological experimental protocols supported by models.

    A protocol defines a specific type of experiment or measurement that a model
    is capable of simulating or fitting.
    """

    FLOW_CURVE = "flow_curve"       # Steady shear viscosity vs shear rate
    CREEP = "creep"                 # Strain vs time at constant stress
    RELAXATION = "relaxation"       # Stress vs time at constant strain
    STARTUP = "startup"             # Stress growth vs time at constant rate
    OSCILLATION = "oscillation"     # Small Amplitude Oscillatory Shear (G', G'')
    LAOS = "laos"                   # Large Amplitude Oscillatory Shear (Lissajous)

    def __str__(self) -> str:
        return self.value


class TransformType(str, Enum):
    """Categories of data transformation operations.

    Transforms are classified by their mathematical operation on the data domain.
    """

    SPECTRAL = "spectral"           # Time <-> Frequency domain (e.g., FFT)
    SUPERPOSITION = "superposition" # Shift data to master curve (e.g., TTS, SRFS)
    DECOMPOSITION = "decomposition" # Split signal into components (e.g., SPP)
    ANALYSIS = "analysis"           # Extract metrics (e.g., Mutation Number, OWChirp)
    PROCESSING = "processing"       # Data cleaning/smoothing (e.g., Smooth Derivative)

    def __str__(self) -> str:
        return self.value
