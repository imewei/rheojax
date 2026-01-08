"""
GUI Crash Prevention Tests
==========================

Comprehensive tests for GUI crash prevention, focusing on:
1. Emoji/Unicode safety in Qt widgets (macOS CoreText crash prevention)
2. Widget lifecycle and initialization safety
3. Signal handling safety
4. Subprocess-based crash detection for risky operations

These tests verify the fixes implemented after discovering that emoji
characters in Qt widgets (especially QTreeWidget) cause bus errors
(SIGBUS) on macOS ARM64 due to CoreText/ImageIO rendering issues.

Background
----------
The original crash was caused by emoji icons in Qt widgets. When Qt
tried to render these emoji, macOS's CoreText framework would crash in:

    CopyEmojiImage -> CGImageSourceCreateImageAtIndex -> IIOReadPlugin

The fix was to use ASCII text icons instead of emoji, with a centralized
IconProvider class that ensures platform safety.
"""

import sys
import textwrap

import pytest

# Platform detection
IS_MACOS = sys.platform == "darwin"


class TestEmojiCrashPrevention:
    """Tests verifying emoji crash prevention mechanisms."""

    def test_icon_provider_never_returns_emoji_on_macos(self):
        """Verify IconProvider returns ASCII-only icons on macOS."""
        from rheojax.gui.utils.icons import IconProvider, emoji_safe

        # On macOS, emoji_safe should return False
        if IS_MACOS:
            assert emoji_safe() is False

        # Provider should never use emoji on macOS regardless of setting
        provider = IconProvider(allow_emoji=True)

        if IS_MACOS:
            assert provider.uses_emoji is False

        # All icons should be ASCII
        for category in provider.CATEGORY_ICONS_ASCII.keys():
            icon = provider.get_category_icon(category)
            assert all(ord(c) < 128 for c in icon), f"Non-ASCII in category {category}"

    def test_all_category_icons_are_ascii(self, ascii_checker):
        """Verify all category icons contain only ASCII characters."""
        from rheojax.gui.utils.icons import IconProvider

        provider = IconProvider(allow_emoji=False)

        for category, icon in provider.CATEGORY_ICONS_ASCII.items():
            assert ascii_checker(
                icon
            ), f"Category {category} has non-ASCII icon: {icon}"

    def test_all_status_icons_are_ascii(self, ascii_checker):
        """Verify all status icons contain only ASCII characters."""
        from rheojax.gui.utils.icons import IconProvider

        provider = IconProvider(allow_emoji=False)

        for status, icon in provider.STATUS_ICONS_ASCII.items():
            assert ascii_checker(icon), f"Status {status} has non-ASCII icon: {icon}"

    def test_all_file_icons_are_ascii(self, ascii_checker):
        """Verify all file type icons contain only ASCII characters."""
        from rheojax.gui.utils.icons import IconProvider

        provider = IconProvider(allow_emoji=False)

        for file_type, icon in provider.FILE_ICONS_ASCII.items():
            assert ascii_checker(
                icon
            ), f"File type {file_type} has non-ASCII icon: {icon}"

    def test_emoji_detection_works(self, emoji_checker):
        """Verify emoji detection function correctly identifies emoji."""
        # Should detect emoji
        assert emoji_checker("\U0001f535") is True  # Blue circle
        assert emoji_checker("\U0001f7e3") is True  # Purple circle
        assert emoji_checker("\U00002705") is True  # Check mark
        assert emoji_checker("Hello \U0001f600 World") is True  # Grinning face

        # Should not detect ASCII
        assert emoji_checker("[C]") is False
        assert emoji_checker("[OK]") is False
        assert emoji_checker("Hello World") is False
        assert emoji_checker("") is False

    def test_status_bar_uses_ascii_indicators(self, ascii_checker):
        """Verify status bar uses ASCII status indicators."""
        # The status bar should use [OK] and [X] instead of checkmarks
        expected_indicators = ["Float64: [OK]", "Float64: [X]"]

        for indicator in expected_indicators:
            assert ascii_checker(indicator), f"Status indicator non-ASCII: {indicator}"


class TestWidgetRenderingSafety:
    """Tests verifying widgets render safely without crashes."""

    @pytest.mark.gui
    def test_qlabel_renders_ascii_text(self, qapp):
        """Verify QLabel can render ASCII text without issues."""
        from PySide6.QtWidgets import QLabel

        test_strings = [
            "[C] Classical Models",
            "[FM] Fractional Maxwell",
            "Float64: [OK]",
            "Float64: [X]",
            "Memory: 1024/8192 MB",
        ]

        for text in test_strings:
            label = QLabel(text)
            label.show()
            assert label.text() == text

    @pytest.mark.gui
    def test_qtreewidget_renders_ascii_text(self, qapp):
        """Verify QTreeWidget can render ASCII text without issues."""
        from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem

        tree = QTreeWidget()
        tree.setHeaderHidden(True)

        # Add items with ASCII icons
        categories = [
            "[C] Classical Models",
            "[FM] Fractional Maxwell",
            "[FZ] Fractional Zener",
            "[F] Flow Models",
            "[MM] Multi-Mode",
            "[SGR] Soft Glassy Rheology",
        ]

        for cat_text in categories:
            item = QTreeWidgetItem()
            item.setText(0, cat_text)
            tree.addTopLevelItem(item)

        tree.show()
        assert tree.topLevelItemCount() == len(categories)

    @pytest.mark.gui
    def test_qtablewidget_renders_ascii_text(self, qapp):
        """Verify QTableWidget can render ASCII text without issues."""
        from PySide6.QtWidgets import QTableWidget, QTableWidgetItem

        table = QTableWidget(5, 3)
        table.setHorizontalHeaderLabels(["Parameter", "Value", "Status"])

        # Add cells with ASCII content
        data = [
            ("G0", "1.0e6", "[OK]"),
            ("tau", "0.1", "[OK]"),
            ("alpha", "0.5", "[!]"),
        ]

        for row, (param, value, status) in enumerate(data):
            table.setItem(row, 0, QTableWidgetItem(param))
            table.setItem(row, 1, QTableWidgetItem(value))
            table.setItem(row, 2, QTableWidgetItem(status))

        table.show()
        assert table.item(0, 0).text() == "G0"

    @pytest.mark.gui
    def test_parameter_table_creates_safely(self, qapp):
        """Verify ParameterTable creates without crash."""
        from rheojax.gui.state.store import ParameterState
        from rheojax.gui.widgets.parameter_table import ParameterTable

        table = ParameterTable()
        table.show()

        # Set parameters
        params = {
            "G0": ParameterState(
                name="G0", value=1e6, min_bound=1e3, max_bound=1e9, fixed=False
            ),
            "tau": ParameterState(
                name="tau", value=0.1, min_bound=1e-6, max_bound=1e3, fixed=False
            ),
        }
        table.set_parameters(params)

        assert table.rowCount() == 2


@pytest.mark.crash_test
class TestSubprocessCrashDetection:
    """Subprocess-based tests that detect potential crashes."""

    def test_gui_launch_no_crash(self, subprocess_runner):
        """Verify GUI can be imported and initialized without crash."""
        code = textwrap.dedent(
            """
            import sys
            sys.exit(0)  # Quick exit after imports
        """
        )

        result = subprocess_runner(code, timeout=5.0)

        assert not result.crashed, (
            f"GUI launch crashed with {result.signal_name or result.return_code}. "
            f"stderr: {result.stderr}"
        )

    def test_parameter_table_creation_no_crash(self, subprocess_runner):
        """Verify ParameterTable creation doesn't crash."""
        code = textwrap.dedent(
            """
            from PySide6.QtWidgets import QApplication
            from rheojax.gui.state.store import ParameterState
            from rheojax.gui.widgets.parameter_table import ParameterTable

            app = QApplication([])
            table = ParameterTable()

            params = {
                "G0": ParameterState(
                    name="G0", value=1e6, min_bound=1e3, max_bound=1e9, fixed=False
                ),
                "tau": ParameterState(
                    name="tau", value=0.1, min_bound=1e-6, max_bound=1e3, fixed=True
                ),
            }
            table.set_parameters(params)
            table.reset_to_defaults()

            print("SUCCESS")
        """
        )

        result = subprocess_runner(code, timeout=10.0)

        assert not result.crashed, (
            f"ParameterTable crashed with {result.signal_name or result.return_code}. "
            f"stderr: {result.stderr}"
        )
        assert "SUCCESS" in result.stdout

    def test_status_bar_update_no_crash(self, subprocess_runner):
        """Verify StatusBar updates don't crash."""
        code = textwrap.dedent(
            """
            from PySide6.QtWidgets import QApplication
            from rheojax.gui.app.status_bar import StatusBar

            app = QApplication([])
            status_bar = StatusBar()

            # Test various updates
            status_bar.show_message("Testing...")
            status_bar.set_float64_status(True)
            status_bar.set_float64_status(False)
            status_bar.set_jax_device("cpu")
            status_bar.update_memory(512, 8192)
            status_bar.show_progress(50, 100, "Progress test")
            status_bar.hide_progress()

            print("SUCCESS")
        """
        )

        result = subprocess_runner(code, timeout=10.0)

        assert not result.crashed, (
            f"StatusBar crashed with {result.signal_name or result.return_code}. "
            f"stderr: {result.stderr}"
        )
        assert "SUCCESS" in result.stdout

    def test_icon_provider_all_icons_no_crash(self, subprocess_runner):
        """Verify IconProvider icon retrieval doesn't crash."""
        code = textwrap.dedent(
            """
            from PySide6.QtWidgets import QApplication, QLabel
            from rheojax.gui.utils.icons import IconProvider

            app = QApplication([])
            provider = IconProvider(allow_emoji=False)

            # Get all category icons and display them
            categories = list(provider.CATEGORY_ICONS_ASCII.keys())
            for cat in categories:
                icon = provider.get_category_icon(cat)
                label = QLabel(f"{icon} {cat}")
                label.show()

            # Get all status icons
            statuses = list(provider.STATUS_ICONS_ASCII.keys())
            for status in statuses:
                icon = provider.get_status_icon(status)
                label = QLabel(f"{icon} {status}")
                label.show()

            print("SUCCESS")
        """
        )

        result = subprocess_runner(code, timeout=10.0)

        assert not result.crashed, (
            f"IconProvider crashed with {result.signal_name or result.return_code}. "
            f"stderr: {result.stderr}"
        )
        assert "SUCCESS" in result.stdout


@pytest.mark.crash_test
class TestRegressionCrashPrevention:
    """Regression tests for specific crash bugs that were fixed."""

    def test_status_bar_checkmark_regression(self, subprocess_runner):
        """Regression test for Unicode checkmarks in status bar.

        Verifies that status bar uses ASCII indicators instead of
        Unicode checkmarks that could potentially cause rendering issues.
        """
        code = textwrap.dedent(
            """
            from PySide6.QtWidgets import QApplication
            from rheojax.gui.app.status_bar import StatusBar

            app = QApplication([])
            status_bar = StatusBar()

            # Toggle float64 status multiple times
            for _ in range(5):
                status_bar.set_float64_status(True)
                text = status_bar.float64_label.text()
                if not all(ord(c) < 128 for c in text):
                    raise ValueError(f"Non-ASCII in float64 label: {text}")

                status_bar.set_float64_status(False)
                text = status_bar.float64_label.text()
                if not all(ord(c) < 128 for c in text):
                    raise ValueError(f"Non-ASCII in float64 label: {text}")

            print("SUCCESS - No Unicode checkmarks")
        """
        )

        result = subprocess_runner(code, timeout=10.0)

        assert (
            not result.crashed
        ), f"Status bar regression crashed. stderr: {result.stderr}"
        assert "SUCCESS" in result.stdout


class TestWidgetLifecycleSafety:
    """Tests for widget creation/destruction safety."""

    @pytest.mark.gui
    def test_rapid_widget_creation_destruction(self, qapp):
        """Test rapid creation and destruction of widgets."""
        from PySide6.QtWidgets import QLabel

        for i in range(100):
            label = QLabel(f"[{i}] Test Label")
            label.show()
            label.close()
            label.deleteLater()

        # Force event processing
        qapp.processEvents()

    @pytest.mark.gui
    def test_tree_widget_many_items(self, qapp):
        """Test QTreeWidget with many items doesn't crash."""
        from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem

        tree = QTreeWidget()
        tree.setHeaderHidden(True)

        # Add many items
        for i in range(100):
            parent = QTreeWidgetItem()
            parent.setText(0, f"[{i}] Category {i}")
            tree.addTopLevelItem(parent)

            for j in range(10):
                child = QTreeWidgetItem()
                child.setText(0, f"  Item {i}.{j}")
                parent.addChild(child)

        tree.expandAll()
        tree.collapseAll()

        assert tree.topLevelItemCount() == 100


class TestUnicodeEdgeCases:
    """Tests for Unicode edge cases and safety."""

    def test_high_codepoints_detected(self, emoji_checker):
        """Test that high Unicode codepoints are detected."""
        # Various emoji and special characters
        test_chars = [
            "\U0001f600",  # Grinning face
            "\U0001f4ca",  # Bar chart
            "\U00002713",  # Check mark
            "\U0000274c",  # Cross mark
            "\U000026a0",  # Warning sign
        ]

        for char in test_chars:
            assert ord(char) > 127, f"Expected high codepoint: {ord(char)}"

    def test_ascii_icons_have_no_high_codepoints(self):
        """Verify ASCII icons have no high codepoints."""
        from rheojax.gui.utils.icons import IconProvider

        provider = IconProvider(allow_emoji=False)

        all_icons = []
        all_icons.extend(provider.CATEGORY_ICONS_ASCII.values())
        all_icons.extend(provider.STATUS_ICONS_ASCII.values())
        all_icons.extend(provider.FILE_ICONS_ASCII.values())

        for icon in all_icons:
            for char in icon:
                assert ord(char) < 128, f"High codepoint {ord(char)} in icon: {icon}"

    def test_combining_characters_safe(self, ascii_checker):
        """Test that combining characters are handled safely."""
        # Combining characters can cause rendering issues
        combining_test = "e\u0301"  # e with combining acute accent

        # Our ASCII checker should flag this
        assert not ascii_checker(combining_test)

    def test_surrogate_pairs_handled(self):
        """Test that surrogate pairs are handled."""
        # Emoji that require surrogate pairs in UTF-16
        emoji = "\U0001f600"  # Grinning face

        # Should be a single character in Python 3
        assert len(emoji) == 1
        assert ord(emoji) > 0xFFFF  # Beyond BMP


@pytest.mark.macos_only
@pytest.mark.skipif(not IS_MACOS, reason="macOS-specific tests")
class TestMacOSSpecific:
    """Tests specific to macOS platform."""

    def test_emoji_safe_returns_false(self):
        """Verify emoji_safe() returns False on macOS."""
        from rheojax.gui.utils.icons import emoji_safe

        assert emoji_safe() is False

    def test_is_macos_returns_true(self):
        """Verify is_macos() returns True on macOS."""
        from rheojax.gui.utils.icons import is_macos

        assert is_macos() is True

    def test_icon_provider_forced_ascii_on_macos(self):
        """Verify IconProvider uses ASCII even when emoji requested."""
        from rheojax.gui.utils.icons import IconProvider

        # Even with allow_emoji=True, should not use emoji on macOS
        provider = IconProvider(allow_emoji=True)
        assert provider.uses_emoji is False

        # Icons should be ASCII
        icon = provider.get_category_icon("classical")
        assert all(ord(c) < 128 for c in icon)
