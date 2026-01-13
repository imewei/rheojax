"""
Platform-Safe Icon Utilities
============================

Centralized icon handling with platform-specific safety for Qt widgets.

macOS ARM64 Emoji Crash Prevention
----------------------------------
Emoji characters in Qt widgets (QTreeWidget, QTableWidget, QListWidget) can
cause bus errors (SIGBUS) on macOS ARM64 due to CoreText/ImageIO rendering
issues. The crash occurs in Apple's ImageIO framework when Qt attempts to
render emoji glyphs:

    CopyEmojiImage -> CGImageSourceCreateImageAtIndex -> IIOReadPlugin::callInitialize

This module provides safe alternatives:
1. ASCII text icons (always safe, cross-platform)
2. Qt standard icons (QStyle.StandardPixmap)
3. Platform-aware emoji (disabled on macOS by default)

Usage
-----
>>> from rheojax.gui.utils.icons import IconProvider
>>> provider = IconProvider()
>>> icon_text = provider.get_category_icon("classical")
>>> # Returns "[C]" on macOS, optionally emoji on other platforms

For Qt icons:
>>> from rheojax.gui.utils.icons import get_standard_icon
>>> icon = get_standard_icon(StandardIcon.FILE)
>>> tree_item.setIcon(0, icon)
"""

import sys
from enum import Enum, auto

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QStyle

from rheojax.logging import get_logger

logger = get_logger(__name__)


class StandardIcon(Enum):
    """Standard Qt icons available cross-platform."""

    FILE = auto()
    FOLDER = auto()
    FOLDER_OPEN = auto()
    SAVE = auto()
    OPEN = auto()
    NEW = auto()
    CLOSE = auto()
    APPLY = auto()
    CANCEL = auto()
    OK = auto()
    HELP = auto()
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()
    QUESTION = auto()
    REFRESH = auto()
    TRASH = auto()
    COMPUTER = auto()
    DRIVE = auto()
    DESKTOP = auto()


# Mapping from StandardIcon to QStyle.StandardPixmap
_ICON_MAPPING: dict[StandardIcon, QStyle.StandardPixmap] = {
    StandardIcon.FILE: QStyle.StandardPixmap.SP_FileIcon,
    StandardIcon.FOLDER: QStyle.StandardPixmap.SP_DirIcon,
    StandardIcon.FOLDER_OPEN: QStyle.StandardPixmap.SP_DirOpenIcon,
    StandardIcon.SAVE: QStyle.StandardPixmap.SP_DialogSaveButton,
    StandardIcon.OPEN: QStyle.StandardPixmap.SP_DialogOpenButton,
    StandardIcon.NEW: QStyle.StandardPixmap.SP_FileIcon,
    StandardIcon.CLOSE: QStyle.StandardPixmap.SP_DialogCloseButton,
    StandardIcon.APPLY: QStyle.StandardPixmap.SP_DialogApplyButton,
    StandardIcon.CANCEL: QStyle.StandardPixmap.SP_DialogCancelButton,
    StandardIcon.OK: QStyle.StandardPixmap.SP_DialogOkButton,
    StandardIcon.HELP: QStyle.StandardPixmap.SP_DialogHelpButton,
    StandardIcon.INFO: QStyle.StandardPixmap.SP_MessageBoxInformation,
    StandardIcon.WARNING: QStyle.StandardPixmap.SP_MessageBoxWarning,
    StandardIcon.CRITICAL: QStyle.StandardPixmap.SP_MessageBoxCritical,
    StandardIcon.QUESTION: QStyle.StandardPixmap.SP_MessageBoxQuestion,
    StandardIcon.REFRESH: QStyle.StandardPixmap.SP_BrowserReload,
    StandardIcon.TRASH: QStyle.StandardPixmap.SP_TrashIcon,
    StandardIcon.COMPUTER: QStyle.StandardPixmap.SP_ComputerIcon,
    StandardIcon.DRIVE: QStyle.StandardPixmap.SP_DriveHDIcon,
    StandardIcon.DESKTOP: QStyle.StandardPixmap.SP_DesktopIcon,
}


def is_macos() -> bool:
    """Check if running on macOS.

    Returns
    -------
    bool
        True if running on macOS (darwin)
    """
    logger.debug("Checking if platform is macOS", platform=sys.platform)
    result = sys.platform == "darwin"
    logger.debug("is_macos check complete", result=result)
    return result


def is_macos_arm64() -> bool:
    """Check if running on macOS ARM64 (Apple Silicon).

    Returns
    -------
    bool
        True if running on macOS with ARM64 architecture
    """
    import platform

    logger.debug(
        "Checking if platform is macOS ARM64",
        platform=sys.platform,
        machine=platform.machine(),
    )
    result = sys.platform == "darwin" and platform.machine() == "arm64"
    logger.debug("is_macos_arm64 check complete", result=result)
    return result


def emoji_safe() -> bool:
    """Check if emoji rendering is safe on current platform.

    Emoji rendering in Qt widgets can cause crashes on macOS due to
    CoreText/ImageIO issues. This function returns False on macOS
    to prevent such crashes.

    Returns
    -------
    bool
        True if emoji can be safely used in Qt widgets
    """
    logger.debug("Checking emoji safety for current platform")
    # Disable emoji on all macOS versions to be safe
    # The CoreText issue affects ARM64 but may also affect Intel
    result = not is_macos()
    logger.debug("emoji_safe check complete", safe=result)
    return result


def get_standard_icon(icon: StandardIcon) -> QIcon:
    """Get a Qt standard icon.

    Parameters
    ----------
    icon : StandardIcon
        The icon type to retrieve

    Returns
    -------
    QIcon
        Qt icon object, or empty icon if not available

    Example
    -------
    >>> icon = get_standard_icon(StandardIcon.FILE)
    >>> tree_item.setIcon(0, icon)
    """
    logger.debug("Getting standard icon", icon=icon.name)
    app = QApplication.instance()
    if app is None:
        logger.debug("No QApplication instance available, returning empty icon")
        return QIcon()

    style = app.style()
    if style is None:
        logger.debug("No QStyle available, returning empty icon")
        return QIcon()

    pixmap = _ICON_MAPPING.get(icon)
    if pixmap is None:
        logger.debug("Icon not found in mapping", icon=icon.name)
        return QIcon()

    logger.debug("Standard icon retrieved successfully", icon=icon.name)
    return style.standardIcon(pixmap)


class IconProvider:
    """Platform-safe icon provider for Qt widgets.

    This class provides text-based icons that are safe to use in Qt widgets
    across all platforms, with optional emoji support on platforms where
    it's safe.

    Parameters
    ----------
    allow_emoji : bool, optional
        Allow emoji on supported platforms (default: False for safety)

    Attributes
    ----------
    CATEGORY_ICONS_ASCII : dict
        ASCII text icons for model categories
    CATEGORY_ICONS_EMOJI : dict
        Emoji icons for model categories (use only when safe)
    STATUS_ICONS_ASCII : dict
        ASCII text icons for status indicators
    STATUS_ICONS_EMOJI : dict
        Emoji icons for status indicators (use only when safe)

    Example
    -------
    >>> provider = IconProvider()
    >>> icon = provider.get_category_icon("classical")
    >>> print(icon)  # "[C]"

    >>> provider_emoji = IconProvider(allow_emoji=True)
    >>> icon = provider_emoji.get_category_icon("classical")
    >>> # Returns emoji on Linux/Windows, ASCII on macOS
    """

    # ASCII text icons - always safe
    CATEGORY_ICONS_ASCII: dict[str, str] = {
        "classical": "[C]",
        "fractional_maxwell": "[FM]",
        "fractional_zener": "[FZ]",
        "fractional_advanced": "[FA]",
        "flow": "[F]",
        "multi_mode": "[MM]",
        "sgr": "[SGR]",
        "other": "[O]",
    }

    # Emoji icons - UNSAFE on macOS, use only when emoji_safe() returns True
    CATEGORY_ICONS_EMOJI: dict[str, str] = {
        "classical": "\U0001f535",  # Blue circle
        "fractional_maxwell": "\U0001f7e3",  # Purple circle
        "fractional_zener": "\U0001f7e0",  # Orange circle
        "fractional_advanced": "\U0001f7e1",  # Yellow circle
        "flow": "\U0001f7e2",  # Green circle
        "multi_mode": "\U0001f7e4",  # Brown circle
        "sgr": "\U0001f534",  # Red circle
        "other": "\U000026aa",  # White circle
    }

    # Status icons - ASCII
    STATUS_ICONS_ASCII: dict[str, str] = {
        "pending": "[ ]",
        "running": "[~]",
        "complete": "[+]",
        "success": "[+]",
        "warning": "[!]",
        "error": "[X]",
        "info": "[i]",
    }

    # Status icons - Emoji (UNSAFE on macOS)
    STATUS_ICONS_EMOJI: dict[str, str] = {
        "pending": "\U000023f3",  # Hourglass
        "running": "\U0001f504",  # Arrows
        "complete": "\U00002705",  # Check mark
        "success": "\U00002705",  # Check mark
        "warning": "\U000026a0\U0000fe0f",  # Warning
        "error": "\U0000274c",  # Cross mark
        "info": "\U00002139\U0000fe0f",  # Info
    }

    # File type icons - ASCII
    FILE_ICONS_ASCII: dict[str, str] = {
        "csv": "[CSV]",
        "excel": "[XLS]",
        "hdf5": "[H5]",
        "json": "[JSON]",
        "folder": "[DIR]",
        "file": "[FILE]",
        "image": "[IMG]",
        "data": "[DAT]",
    }

    # File type icons - Emoji (UNSAFE on macOS)
    FILE_ICONS_EMOJI: dict[str, str] = {
        "csv": "\U0001f4c4",  # Page
        "excel": "\U0001f4ca",  # Chart
        "hdf5": "\U0001f4be",  # Floppy
        "json": "\U0001f4dd",  # Memo
        "folder": "\U0001f4c1",  # Folder
        "file": "\U0001f4c4",  # Page
        "image": "\U0001f5bc\U0000fe0f",  # Frame
        "data": "\U0001f4c8",  # Chart
    }

    def __init__(self, allow_emoji: bool = False) -> None:
        """Initialize icon provider.

        Parameters
        ----------
        allow_emoji : bool, optional
            Allow emoji on supported platforms (default: False)
        """
        logger.debug("Initializing IconProvider", allow_emoji=allow_emoji)
        self._allow_emoji = allow_emoji and emoji_safe()
        logger.debug(
            "IconProvider initialized",
            allow_emoji_requested=allow_emoji,
            emoji_enabled=self._allow_emoji,
        )

    @property
    def uses_emoji(self) -> bool:
        """Check if this provider uses emoji icons.

        Returns
        -------
        bool
            True if emoji icons are enabled and safe
        """
        return self._allow_emoji

    def get_category_icon(self, category: str) -> str:
        """Get icon for a model category.

        Parameters
        ----------
        category : str
            Category name (e.g., "classical", "flow", "sgr")

        Returns
        -------
        str
            Icon text (ASCII or emoji based on settings)
        """
        logger.debug(
            "Getting category icon", category=category, uses_emoji=self._allow_emoji
        )
        if self._allow_emoji:
            icon = self.CATEGORY_ICONS_EMOJI.get(
                category, self.CATEGORY_ICONS_EMOJI.get("other", "[?]")
            )
        else:
            icon = self.CATEGORY_ICONS_ASCII.get(
                category, self.CATEGORY_ICONS_ASCII.get("other", "[?]")
            )
        logger.debug("Category icon retrieved", category=category, icon=icon)
        return icon

    def get_status_icon(self, status: str) -> str:
        """Get icon for a status indicator.

        Parameters
        ----------
        status : str
            Status name (e.g., "pending", "complete", "error")

        Returns
        -------
        str
            Icon text (ASCII or emoji based on settings)
        """
        logger.debug("Getting status icon", status=status, uses_emoji=self._allow_emoji)
        if self._allow_emoji:
            icon = self.STATUS_ICONS_EMOJI.get(status, "[?]")
        else:
            icon = self.STATUS_ICONS_ASCII.get(status, "[?]")
        logger.debug("Status icon retrieved", status=status, icon=icon)
        return icon

    def get_file_icon(self, file_type: str) -> str:
        """Get icon for a file type.

        Parameters
        ----------
        file_type : str
            File type (e.g., "csv", "excel", "folder")

        Returns
        -------
        str
            Icon text (ASCII or emoji based on settings)
        """
        logger.debug(
            "Getting file icon", file_type=file_type, uses_emoji=self._allow_emoji
        )
        if self._allow_emoji:
            icon = self.FILE_ICONS_EMOJI.get(file_type, "[?]")
        else:
            icon = self.FILE_ICONS_ASCII.get(file_type, "[?]")
        logger.debug("File icon retrieved", file_type=file_type, icon=icon)
        return icon

    def format_with_icon(
        self,
        text: str,
        icon_type: str,
        icon_category: str = "status",
    ) -> str:
        """Format text with a prepended icon.

        Parameters
        ----------
        text : str
            Text to format
        icon_type : str
            Icon type name
        icon_category : str, optional
            Icon category: "status", "category", or "file"

        Returns
        -------
        str
            Formatted string with icon

        Example
        -------
        >>> provider = IconProvider()
        >>> provider.format_with_icon("Fitting complete", "success", "status")
        "[+] Fitting complete"
        """
        logger.debug(
            "Formatting text with icon",
            text=text,
            icon_type=icon_type,
            icon_category=icon_category,
        )
        if icon_category == "status":
            icon = self.get_status_icon(icon_type)
        elif icon_category == "category":
            icon = self.get_category_icon(icon_type)
        elif icon_category == "file":
            icon = self.get_file_icon(icon_type)
        else:
            logger.debug(
                "Unknown icon category, using fallback", icon_category=icon_category
            )
            icon = "[?]"

        result = f"{icon} {text}"
        logger.debug("Text formatted with icon", result=result)
        return result


# Module-level singleton for convenience
_default_provider: IconProvider | None = None


def get_icon_provider(allow_emoji: bool = False) -> IconProvider:
    """Get the default icon provider singleton.

    Parameters
    ----------
    allow_emoji : bool, optional
        Allow emoji on supported platforms

    Returns
    -------
    IconProvider
        Icon provider instance
    """
    global _default_provider
    logger.debug("Getting icon provider singleton", allow_emoji=allow_emoji)
    if _default_provider is None or _default_provider._allow_emoji != (
        allow_emoji and emoji_safe()
    ):
        logger.debug("Creating new IconProvider singleton", allow_emoji=allow_emoji)
        _default_provider = IconProvider(allow_emoji=allow_emoji)
    else:
        logger.debug("Returning existing IconProvider singleton")
    return _default_provider
