"""
Configuration Management
=======================

Application configuration and persistence.
"""

from pathlib import Path
from typing import Any


class Config:
    """Application configuration manager.

    Features:
        - Persistent settings
        - Default values
        - Validation
        - Config file I/O

    Example
    -------
    >>> config = Config()  # doctest: +SKIP
    >>> config.set('theme', 'dark')  # doctest: +SKIP
    >>> theme = config.get('theme', default='light')  # doctest: +SKIP
    """

    def __init__(self, config_path: Path | None = None) -> None:
        """Initialize configuration manager.

        Parameters
        ----------
        config_path : Path, optional
            Path to config file
        """
        ...

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.

        Parameters
        ----------
        key : str
            Configuration key
        default : Any, optional
            Default value if not found

        Returns
        -------
        Any
            Configuration value
        """
        ...

    def set(self, key: str, value: Any) -> None:
        """Set configuration value.

        Parameters
        ----------
        key : str
            Configuration key
        value : Any
            Configuration value
        """
        ...

    def save(self) -> None:
        """Save configuration to file."""
        ...

    def load(self) -> None:
        """Load configuration from file."""
        ...

    def reset(self) -> None:
        """Reset to default configuration."""
        ...
