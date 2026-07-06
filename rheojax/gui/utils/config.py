"""
Configuration Management
=======================

Application configuration and persistence.
"""

import json
from pathlib import Path
from typing import Any

from rheojax.logging import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG_PATH = Path.home() / ".rheojax" / "gui_config.json"


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
        logger.debug(
            "Initializing Config manager",
            config_path=str(config_path) if config_path else None,
        )
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self._data: dict[str, Any] = {}
        self.load()

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
        logger.debug("Getting config value", key=key, has_default=default is not None)
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value.

        Parameters
        ----------
        key : str
            Configuration key
        value : Any
            Configuration value
        """
        logger.debug("Setting config value", key=key, value_type=type(value).__name__)
        self._data[key] = value

    def save(self) -> None:
        """Save configuration to file."""
        logger.debug("Saving configuration to file")
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self.config_path.write_text(json.dumps(self._data, indent=2))
        except (OSError, TypeError, ValueError) as exc:
            logger.warning(
                "Failed to save config",
                config_path=str(self.config_path),
                error=str(exc),
            )

    def load(self) -> None:
        """Load configuration from file."""
        logger.debug("Loading configuration from file")
        if not self.config_path.exists():
            self._data = {}
            return
        try:
            loaded = json.loads(self.config_path.read_text())
            if not isinstance(loaded, dict):
                raise ValueError(
                    f"Config root must be an object, got {type(loaded).__name__}"
                )
            self._data = loaded
        except (OSError, ValueError) as exc:
            logger.warning(
                "Failed to load config, using empty config",
                config_path=str(self.config_path),
                error=str(exc),
            )
            self._data = {}

    def reset(self) -> None:
        """Reset to default configuration."""
        logger.debug("Resetting configuration to defaults")
        self._data = {}
