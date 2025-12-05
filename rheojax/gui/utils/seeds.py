"""
Seed Management
==============

Random seed management for reproducibility.
"""

from typing import Any

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()


class SeedManager:
    """Random seed manager for reproducibility.

    Features:
        - JAX PRNG key management
        - Seed tracking
        - Reproducible workflows

    Example
    -------
    >>> manager = SeedManager(seed=42)  # doctest: +SKIP
    >>> key = manager.get_key()  # doctest: +SKIP
    """

    def __init__(self, seed: int | None = None) -> None:
        """Initialize seed manager.

        Parameters
        ----------
        seed : int, optional
            Initial random seed
        """
        ...

    def get_key(self) -> Any:
        """Get current PRNG key.

        Returns
        -------
        PRNGKey
            JAX random key
        """
        ...

    def split_key(self, num: int = 2) -> list[Any]:
        """Split PRNG key.

        Parameters
        ----------
        num : int, default=2
            Number of keys to generate

        Returns
        -------
        list[PRNGKey]
            Split keys
        """
        ...

    def set_seed(self, seed: int) -> None:
        """Set new random seed.

        Parameters
        ----------
        seed : int
            Random seed
        """
        ...

    def get_seed(self) -> int | None:
        """Get current seed.

        Returns
        -------
        int | None
            Current seed
        """
        ...
