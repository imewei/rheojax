"""
Tests for Platform-Safe Icon Utilities
======================================

Tests for rheojax.gui.utils.icons module which provides
platform-safe icon handling to prevent macOS CoreText crashes.
"""

import sys
from unittest.mock import patch

import pytest


class TestPlatformDetection:
    """Tests for platform detection functions."""

    def test_is_macos_on_darwin(self):
        """Test is_macos returns True on darwin platform."""
        from rheojax.gui.utils.icons import is_macos

        with patch.object(sys, "platform", "darwin"):
            assert is_macos() is True

    def test_is_macos_on_linux(self):
        """Test is_macos returns False on linux platform."""
        from rheojax.gui.utils.icons import is_macos

        with patch.object(sys, "platform", "linux"):
            assert is_macos() is False

    def test_is_macos_on_windows(self):
        """Test is_macos returns False on windows platform."""
        from rheojax.gui.utils.icons import is_macos

        with patch.object(sys, "platform", "win32"):
            assert is_macos() is False

    def test_emoji_safe_false_on_macos(self):
        """Test emoji_safe returns False on macOS."""
        from rheojax.gui.utils.icons import emoji_safe

        with patch.object(sys, "platform", "darwin"):
            assert emoji_safe() is False

    def test_emoji_safe_true_on_linux(self):
        """Test emoji_safe returns True on Linux."""
        from rheojax.gui.utils.icons import emoji_safe

        with patch.object(sys, "platform", "linux"):
            assert emoji_safe() is True

    def test_emoji_safe_true_on_windows(self):
        """Test emoji_safe returns True on Windows."""
        from rheojax.gui.utils.icons import emoji_safe

        with patch.object(sys, "platform", "win32"):
            assert emoji_safe() is True


class TestIconProvider:
    """Tests for IconProvider class."""

    def test_default_provider_uses_ascii(self):
        """Test default provider uses ASCII icons."""
        from rheojax.gui.utils.icons import IconProvider

        provider = IconProvider()
        assert provider.uses_emoji is False

    def test_provider_with_emoji_false_uses_ascii(self):
        """Test provider with allow_emoji=False uses ASCII."""
        from rheojax.gui.utils.icons import IconProvider

        provider = IconProvider(allow_emoji=False)
        assert provider.uses_emoji is False

    def test_provider_category_icons_are_ascii(self):
        """Test category icons are ASCII strings without emoji."""
        from rheojax.gui.utils.icons import IconProvider

        provider = IconProvider(allow_emoji=False)

        categories = [
            "classical",
            "fractional_maxwell",
            "fractional_zener",
            "fractional_advanced",
            "flow",
            "multi_mode",
            "sgr",
            "other",
        ]

        for category in categories:
            icon = provider.get_category_icon(category)
            # Check it's ASCII (no characters above 127)
            assert all(ord(c) < 128 for c in icon), f"Non-ASCII in {category}: {icon}"
            # Check it's bracketed format
            assert icon.startswith("[") and icon.endswith("]")

    def test_provider_status_icons_are_ascii(self):
        """Test status icons are ASCII strings without emoji."""
        from rheojax.gui.utils.icons import IconProvider

        provider = IconProvider(allow_emoji=False)

        statuses = ["pending", "running", "complete", "success", "warning", "error", "info"]

        for status in statuses:
            icon = provider.get_status_icon(status)
            # Check it's ASCII
            assert all(ord(c) < 128 for c in icon), f"Non-ASCII in {status}: {icon}"
            # Check it's bracketed format
            assert icon.startswith("[") and icon.endswith("]")

    def test_provider_file_icons_are_ascii(self):
        """Test file type icons are ASCII strings without emoji."""
        from rheojax.gui.utils.icons import IconProvider

        provider = IconProvider(allow_emoji=False)

        file_types = ["csv", "excel", "hdf5", "json", "folder", "file", "image", "data"]

        for file_type in file_types:
            icon = provider.get_file_icon(file_type)
            # Check it's ASCII
            assert all(ord(c) < 128 for c in icon), f"Non-ASCII in {file_type}: {icon}"
            # Check it's bracketed format
            assert icon.startswith("[") and icon.endswith("]")

    def test_provider_unknown_category_returns_other(self):
        """Test unknown category returns 'other' icon."""
        from rheojax.gui.utils.icons import IconProvider

        provider = IconProvider(allow_emoji=False)
        icon = provider.get_category_icon("nonexistent_category")
        expected = provider.get_category_icon("other")
        assert icon == expected

    def test_provider_format_with_icon(self):
        """Test format_with_icon combines icon and text."""
        from rheojax.gui.utils.icons import IconProvider

        provider = IconProvider(allow_emoji=False)

        result = provider.format_with_icon("Test message", "success", "status")
        assert result.startswith("[")
        assert "Test message" in result

    def test_emoji_disabled_on_macos_even_when_requested(self):
        """Test emoji is disabled on macOS even when allow_emoji=True."""
        from rheojax.gui.utils.icons import IconProvider

        with patch.object(sys, "platform", "darwin"):
            provider = IconProvider(allow_emoji=True)
            # Should still be False because macOS is not emoji-safe
            assert provider.uses_emoji is False

    def test_emoji_enabled_on_linux_when_requested(self):
        """Test emoji is enabled on Linux when allow_emoji=True."""
        from rheojax.gui.utils.icons import IconProvider

        with patch.object(sys, "platform", "linux"):
            provider = IconProvider(allow_emoji=True)
            assert provider.uses_emoji is True


class TestGetIconProvider:
    """Tests for get_icon_provider singleton function."""

    def test_get_icon_provider_returns_provider(self):
        """Test get_icon_provider returns an IconProvider."""
        from rheojax.gui.utils.icons import IconProvider, get_icon_provider

        provider = get_icon_provider()
        assert isinstance(provider, IconProvider)

    def test_get_icon_provider_default_no_emoji(self):
        """Test get_icon_provider default does not use emoji."""
        from rheojax.gui.utils.icons import get_icon_provider

        provider = get_icon_provider(allow_emoji=False)
        assert provider.uses_emoji is False


class TestStandardIcon:
    """Tests for StandardIcon enum."""

    def test_standard_icon_enum_values(self):
        """Test StandardIcon enum has expected values."""
        from rheojax.gui.utils.icons import StandardIcon

        expected_icons = [
            "FILE",
            "FOLDER",
            "FOLDER_OPEN",
            "SAVE",
            "OPEN",
            "CLOSE",
            "OK",
            "CANCEL",
            "HELP",
            "INFO",
            "WARNING",
            "CRITICAL",
        ]

        for name in expected_icons:
            assert hasattr(StandardIcon, name), f"Missing StandardIcon.{name}"


class TestModelBrowserIntegration:
    """Integration tests for model_browser using icons."""

    def test_model_browser_get_category_icon(self):
        """Test model_browser get_category_icon function."""
        from rheojax.gui.widgets.model_browser import get_category_icon

        # All categories should return ASCII icons
        categories = [
            "classical",
            "fractional_maxwell",
            "fractional_zener",
            "fractional_advanced",
            "flow",
            "multi_mode",
            "sgr",
            "other",
        ]

        for category in categories:
            icon = get_category_icon(category)
            # Verify ASCII only
            assert all(ord(c) < 128 for c in icon), f"Non-ASCII icon for {category}"

    def test_model_browser_category_info_no_emoji(self):
        """Test CATEGORY_INFO dict has no emoji in values."""
        from rheojax.gui.widgets.model_browser import CATEGORY_INFO

        for category, info in CATEGORY_INFO.items():
            for key, value in info.items():
                if isinstance(value, str):
                    # Check no emoji (characters above U+1F000 are typically emoji)
                    has_emoji = any(ord(c) > 0x1F000 for c in value)
                    assert not has_emoji, f"Emoji found in CATEGORY_INFO[{category}][{key}]"


class TestNoEmojiInQtWidgets:
    """Tests to ensure no emoji leaks into Qt widget text."""

    def test_category_icons_safe_for_qtreewidget(self):
        """Test all category icons are safe for QTreeWidget rendering."""
        from rheojax.gui.utils.icons import IconProvider

        provider = IconProvider(allow_emoji=False)

        # These are the categories used in QTreeWidget
        categories = list(provider.CATEGORY_ICONS_ASCII.keys())

        for category in categories:
            icon = provider.get_category_icon(category)

            # Must be ASCII only
            assert all(ord(c) < 128 for c in icon)

            # Must not contain any Unicode emoji ranges
            # Emoji range: U+1F300 to U+1FAFF (and others)
            for char in icon:
                code = ord(char)
                assert not (0x1F300 <= code <= 0x1FAFF), f"Emoji codepoint in {category}"
                assert not (0x2600 <= code <= 0x26FF), f"Misc symbol in {category}"
                assert not (0x2700 <= code <= 0x27BF), f"Dingbat in {category}"

    def test_status_icons_safe_for_qtablewidget(self):
        """Test all status icons are safe for QTableWidget rendering."""
        from rheojax.gui.utils.icons import IconProvider

        provider = IconProvider(allow_emoji=False)

        statuses = list(provider.STATUS_ICONS_ASCII.keys())

        for status in statuses:
            icon = provider.get_status_icon(status)

            # Must be ASCII only
            assert all(ord(c) < 128 for c in icon)
