"""Multi-mode rheological models.

Contains models with multiple relaxation modes:
- GeneralizedMaxwell: Generalized Maxwell Model (Prony series, N modes)
"""

from rheojax.models.multimode.generalized_maxwell import GeneralizedMaxwell

__all__ = ["GeneralizedMaxwell"]
