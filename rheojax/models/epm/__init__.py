"""Elasto-Plastic Models (EPM) for amorphous solids."""

from .lattice import LatticeEPM
from .tensor import TensorialEPM

__all__ = ["LatticeEPM", "TensorialEPM"]
