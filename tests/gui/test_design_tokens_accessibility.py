"""Tests for design token migration and keyboard accessibility.

Validates that:
- Pages use design tokens instead of hardcoded values
- Clickable widgets have keyboard accessibility (focus policy, key handlers)
- Import consistency (no direct PySide6 imports in pages)
"""

import re

import pytest

pytestmark = pytest.mark.gui

try:
    from PySide6.QtWidgets import QApplication

    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False


# =============================================================================
# Design Token Migration Tests
# =============================================================================


class TestDesignTokenUsage:
    """Verify pages reference design tokens instead of hardcoded values."""

    @pytest.mark.smoke
    def test_home_page_no_raw_hex_in_stylesheets(self) -> None:
        """home_page.py should not contain bare hex colour literals
        outside of gradient stop definitions that already use token refs.
        """
        import inspect

        from rheojax.gui.pages.home_page import HomePage

        source = inspect.getsource(HomePage)

        # These hardcoded hex colors should no longer appear
        banned = ["#0F172A", "#1E3A8A", "#4338CA", "#94A3B8", "#E2E8F0", "#64748B"]
        for color in banned:
            assert color not in source, (
                f"Hardcoded color {color} found in HomePage source — "
                "should use ColorPalette token"
            )

    @pytest.mark.smoke
    def test_home_page_uses_typography_tokens(self) -> None:
        """home_page.py should reference Typography tokens for fonts."""
        import inspect

        from rheojax.gui.pages.home_page import HomePage

        source = inspect.getsource(HomePage)

        # The old hardcoded font-family string should be gone
        assert "'Inter'" not in source, (
            "Hardcoded 'Inter' font-family found — should use Typography.FONT_FAMILY"
        )

    @pytest.mark.smoke
    def test_data_page_uses_typography_tokens(self) -> None:
        """data_page.py should use Typography for font-size/font-weight."""
        import inspect

        from rheojax.gui.pages.data_page import DataPage

        source = inspect.getsource(DataPage)

        # Check that bare font-size: 24pt was migrated
        assert "font-size: 24pt" not in source, (
            "Hardcoded 'font-size: 24pt' found — should use Typography.SIZE_HEADING"
        )

    @pytest.mark.smoke
    def test_transform_page_no_hardcoded_white(self) -> None:
        """transform_page.py should not have bare 'color: white' inline."""
        import inspect

        from rheojax.gui.pages.transform_page import TransformPage

        source = inspect.getsource(TransformPage)
        # The old "color: white; font-size: 13pt; font-weight: bold;"
        assert "color: white;" not in source, (
            "Hardcoded 'color: white' found — should use ColorPalette.TEXT_INVERSE"
        )

    @pytest.mark.smoke
    def test_transform_page_configure_button_uses_button_style(self) -> None:
        """Configure button should use button_style() helper."""
        import inspect

        from rheojax.gui.pages.transform_page import TransformPage

        source = inspect.getsource(TransformPage)
        assert "background-color: white; color: black" not in source, (
            "Hardcoded button style found — should use button_style()"
        )

    @pytest.mark.smoke
    def test_export_page_uses_spacing_tokens(self) -> None:
        """export_page.py should use Spacing tokens for margin-top."""
        import inspect

        from rheojax.gui.pages.export_page import ExportPage

        source = inspect.getsource(ExportPage)
        # The old hardcoded 'margin-top: 15px' should be gone
        assert "margin-top: 15px" not in source, (
            "Hardcoded 'margin-top: 15px' found — should use Spacing.LG"
        )

    @pytest.mark.smoke
    def test_diagnostics_page_no_direct_pyside6_import(self) -> None:
        """diagnostics_page.py should not import from PySide6 directly."""
        import inspect

        from rheojax.gui.pages.diagnostics_page import DiagnosticsPage

        source = inspect.getsource(DiagnosticsPage)
        assert "from PySide6" not in source, (
            "Direct PySide6 import found — should use rheojax.gui.compat"
        )

    @pytest.mark.smoke
    def test_home_page_chart_colors_use_tokens(self) -> None:
        """Example card chart colors should reference ColorPalette tokens."""
        import inspect

        from rheojax.gui.pages.home_page import HomePage

        source = inspect.getsource(HomePage)
        # Old hardcoded chart list included #0891B2 (cyan) and #DB2777 (pink)
        assert "#0891B2" not in source, (
            "Hardcoded chart color #0891B2 found — "
            "should use ColorPalette.CHART_* or similar"
        )


# =============================================================================
# Keyboard Accessibility Tests
# =============================================================================


@pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 not installed")
class TestKeyboardAccessibility:
    """Verify interactive widgets have keyboard support."""

    @pytest.fixture(autouse=True)
    def _ensure_qapp(self, qapp):
        """Ensure QApplication exists for widget tests."""

    @pytest.mark.smoke
    def test_workflow_cards_are_focusable(self) -> None:
        """Workflow cards should accept keyboard focus."""
        from rheojax.gui.compat import Qt
        from rheojax.gui.pages.home_page import HomePage

        page = HomePage()
        # Find QFrame children with "card-clickable" property
        from rheojax.gui.compat import QFrame

        cards = [
            w
            for w in page.findChildren(QFrame)
            if w.property("class") == "card-clickable"
        ]
        assert len(cards) >= 2, "Should have at least 2 workflow cards"
        for card in cards:
            assert card.focusPolicy() == Qt.StrongFocus, (
                f"Workflow card should have StrongFocus policy, got {card.focusPolicy()}"
            )

    @pytest.mark.smoke
    def test_workflow_cards_have_accessible_names(self) -> None:
        """Workflow cards should have accessible names for screen readers."""
        from rheojax.gui.compat import QFrame
        from rheojax.gui.pages.home_page import HomePage

        page = HomePage()
        cards = [
            w
            for w in page.findChildren(QFrame)
            if w.property("class") == "card-clickable"
        ]
        for card in cards:
            name = card.accessibleName()
            assert name, "Workflow card should have an accessible name"
            assert "workflow" in name.lower(), (
                f"Accessible name '{name}' should mention 'workflow'"
            )

    @pytest.mark.smoke
    def test_example_cards_are_focusable(self) -> None:
        """Example dataset cards should accept keyboard focus."""
        from rheojax.gui.compat import Qt, QWidget
        from rheojax.gui.pages.home_page import HomePage

        page = HomePage()
        # Example cards have accessible names starting with "Example dataset:"
        focusable_examples = [
            w
            for w in page.findChildren(QWidget)
            if w.accessibleName().startswith("Example dataset:")
        ]
        # There should be 8 example cards
        assert len(focusable_examples) == 8, (
            f"Expected 8 focusable example cards, found {len(focusable_examples)}"
        )
        for card in focusable_examples:
            assert card.focusPolicy() == Qt.StrongFocus

    @pytest.mark.smoke
    def test_drop_zone_is_focusable(self) -> None:
        """DropZone should accept keyboard focus."""
        from rheojax.gui.compat import Qt
        from rheojax.gui.pages.data_page import DropZone

        drop_zone = DropZone()
        assert drop_zone.focusPolicy() == Qt.StrongFocus
        assert drop_zone.accessibleName() == "File drop zone"


# =============================================================================
# QSS / Card Component Tests
# =============================================================================


@pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 not installed")
class TestCardQSSCompatibility:
    """Verify card widgets match QSS selector requirements."""

    @pytest.fixture(autouse=True)
    def _ensure_qapp(self, qapp):
        """Ensure QApplication exists for widget tests."""

    @pytest.mark.smoke
    def test_workflow_cards_have_styled_panel_frame_shape(self) -> None:
        """Workflow cards must have frameShape=StyledPanel for QSS matching."""
        from rheojax.gui.compat import QFrame
        from rheojax.gui.pages.home_page import HomePage

        page = HomePage()
        cards = [
            w
            for w in page.findChildren(QFrame)
            if w.property("class") == "card-clickable"
        ]
        assert len(cards) >= 2, "Should have at least 2 workflow cards"
        for card in cards:
            assert card.frameShape() == QFrame.StyledPanel, (
                "Card must have StyledPanel frameShape for QSS selector "
                'QFrame[frameShape="StyledPanel"][class="card-clickable"] to match'
            )

    @pytest.mark.smoke
    def test_workflow_cards_use_clickable_frame_subclass(self) -> None:
        """Workflow cards should be ClickableFrame instances (not monkey-patched)."""
        from rheojax.gui.compat import QFrame
        from rheojax.gui.pages.home_page import ClickableFrame, HomePage

        page = HomePage()
        cards = [
            w
            for w in page.findChildren(QFrame)
            if w.property("class") == "card-clickable"
        ]
        for card in cards:
            assert isinstance(card, ClickableFrame), (
                f"Card should be ClickableFrame, got {type(card).__name__}"
            )


# =============================================================================
# Token-QSS Synchronization Tests
# =============================================================================


class TestTokenQSSSync:
    """Verify design tokens are synchronized with QSS values."""

    @pytest.mark.smoke
    def test_primary_color_is_indigo(self) -> None:
        """ColorPalette.PRIMARY should be indigo-700 (#4338CA) matching QSS."""
        from rheojax.gui.resources.styles.tokens import ColorPalette

        assert ColorPalette.PRIMARY == "#4338CA", (
            f"PRIMARY should be #4338CA (indigo-700), got {ColorPalette.PRIMARY}"
        )

    @pytest.mark.smoke
    def test_font_family_starts_with_inter(self) -> None:
        """Typography.FONT_FAMILY should start with Inter matching QSS."""
        from rheojax.gui.resources.styles.tokens import Typography

        assert Typography.FONT_FAMILY.startswith('"Inter"'), (
            f"FONT_FAMILY should start with Inter, got: {Typography.FONT_FAMILY[:30]}"
        )

    @pytest.mark.smoke
    def test_size_md_sm_exists(self) -> None:
        """Typography.SIZE_MD_SM should be 11 (between SM=10 and BASE=12)."""
        from rheojax.gui.resources.styles.tokens import Typography

        assert Typography.SIZE_MD_SM == 11
        assert Typography.SIZE_SM < Typography.SIZE_MD_SM < Typography.SIZE_BASE

    @pytest.mark.smoke
    def test_size_hero_exists(self) -> None:
        """Typography.SIZE_HERO should exist for hero/display titles."""
        from rheojax.gui.resources.styles.tokens import Typography

        assert Typography.SIZE_HERO == 36
        assert Typography.SIZE_HERO > Typography.SIZE_HEADING


# =============================================================================
# Theme Manager Tests
# =============================================================================


class TestThemeManager:
    """Verify ThemeManager provides correct palette per theme."""

    @pytest.mark.smoke
    def test_default_theme_is_light(self) -> None:
        """ThemeManager defaults to light theme."""
        from rheojax.gui.resources.styles.tokens import ThemeManager

        # Reset to known state
        ThemeManager.set_theme("light")
        assert not ThemeManager.is_dark()

    @pytest.mark.smoke
    def test_set_dark_theme(self) -> None:
        """ThemeManager.set_theme('dark') switches to dark palette."""
        from rheojax.gui.resources.styles.tokens import ColorPalette, ThemeManager

        ThemeManager.set_theme("dark")
        assert ThemeManager.is_dark()
        # Dark palette should NOT be the light ColorPalette
        palette = ThemeManager.get_palette()
        assert palette is not ColorPalette
        assert palette.BG_BASE == "#0F172A"  # slate-900
        # Clean up
        ThemeManager.set_theme("light")

    @pytest.mark.smoke
    def test_set_light_theme(self) -> None:
        """ThemeManager.set_theme('light') returns light palette."""
        from rheojax.gui.resources.styles.tokens import ColorPalette, ThemeManager

        ThemeManager.set_theme("light")
        palette = ThemeManager.get_palette()
        assert palette is ColorPalette
        assert palette.BG_BASE == "#FFFFFF"

    @pytest.mark.smoke
    def test_themed_returns_light_by_default(self) -> None:
        """themed() returns light palette values when theme is light."""
        from rheojax.gui.resources.styles.tokens import (
            ColorPalette,
            ThemeManager,
            themed,
        )

        ThemeManager.set_theme("light")
        assert themed("PRIMARY") == ColorPalette.PRIMARY
        assert themed("BG_BASE") == "#FFFFFF"

    @pytest.mark.smoke
    def test_themed_returns_dark_when_dark(self) -> None:
        """themed() returns dark palette values when theme is dark."""
        from rheojax.gui.resources.styles.tokens import (
            DarkColorPalette,
            ThemeManager,
            themed,
        )

        ThemeManager.set_theme("dark")
        assert themed("PRIMARY") == DarkColorPalette.PRIMARY
        assert themed("BG_BASE") == "#0F172A"
        # Clean up
        ThemeManager.set_theme("light")


class TestDarkColorPaletteParity:
    """Verify DarkColorPalette has the same attributes as ColorPalette."""

    @pytest.mark.smoke
    def test_all_light_tokens_exist_in_dark(self) -> None:
        """Every ClassVar in ColorPalette must also exist in DarkColorPalette."""
        from rheojax.gui.resources.styles.tokens import ColorPalette, DarkColorPalette

        light_attrs = {
            k
            for k in dir(ColorPalette)
            if not k.startswith("_") and isinstance(getattr(ColorPalette, k), str)
        }
        dark_attrs = {
            k
            for k in dir(DarkColorPalette)
            if not k.startswith("_") and isinstance(getattr(DarkColorPalette, k), str)
        }

        missing = light_attrs - dark_attrs
        assert not missing, (
            f"DarkColorPalette is missing attributes present in ColorPalette: {missing}"
        )

    @pytest.mark.smoke
    def test_dark_palette_values_are_valid_hex(self) -> None:
        """All DarkColorPalette values should be valid hex color strings."""
        from rheojax.gui.resources.styles.tokens import DarkColorPalette

        hex_pattern = re.compile(r"^#[0-9A-Fa-f]{6}$")
        for attr in dir(DarkColorPalette):
            if attr.startswith("_"):
                continue
            val = getattr(DarkColorPalette, attr)
            if isinstance(val, str):
                assert hex_pattern.match(val), (
                    f"DarkColorPalette.{attr} = '{val}' is not a valid hex color"
                )

    @pytest.mark.smoke
    def test_bright_variants_exist(self) -> None:
        """Both palettes should have SUCCESS/WARNING/ERROR/ACCENT_BRIGHT."""
        from rheojax.gui.resources.styles.tokens import ColorPalette, DarkColorPalette

        for name in (
            "SUCCESS_BRIGHT",
            "WARNING_BRIGHT",
            "ERROR_BRIGHT",
            "ACCENT_BRIGHT",
        ):
            assert hasattr(ColorPalette, name), f"ColorPalette missing {name}"
            assert hasattr(DarkColorPalette, name), f"DarkColorPalette missing {name}"


class TestNoSizeBaseArithmetic:
    """Verify SIZE_BASE - 1 arithmetic has been replaced."""

    @pytest.mark.smoke
    def test_data_page_no_size_base_minus_one(self) -> None:
        """data_page.py should use SIZE_MD_SM instead of SIZE_BASE - 1."""
        import inspect

        from rheojax.gui.pages.data_page import DataPage

        source = inspect.getsource(DataPage)
        assert "SIZE_BASE - 1" not in source, (
            "SIZE_BASE - 1 found — should use Typography.SIZE_MD_SM"
        )

    @pytest.mark.smoke
    def test_transform_page_no_size_base_minus_one(self) -> None:
        """transform_page.py should use SIZE_MD_SM instead of SIZE_BASE - 1."""
        import inspect

        from rheojax.gui.pages.transform_page import TransformPage

        source = inspect.getsource(TransformPage)
        assert "SIZE_BASE - 1" not in source, (
            "SIZE_BASE - 1 found — should use Typography.SIZE_MD_SM"
        )


class TestHomePageGradientTokens:
    """Verify home_page gradient hex colors have been replaced with tokens."""

    @pytest.mark.smoke
    def test_no_hardcoded_gradient_hex(self) -> None:
        """Gradient hex values should be replaced with themed() token calls."""
        import inspect

        from rheojax.gui.pages.home_page import HomePage

        source = inspect.getsource(HomePage)
        # These were the old hardcoded gradient hex values
        banned_gradient_hex = [
            "#10B981",
            "#F59E0B",
            "#EF4444",
            "#8B5CF6",
        ]
        for color in banned_gradient_hex:
            assert color not in source, (
                f"Hardcoded gradient color {color} found — should use themed() token"
            )

    @pytest.mark.smoke
    def test_no_hardcoded_px_font_sizes(self) -> None:
        """home_page.py should not use hardcoded px font sizes in inline styles."""
        import inspect

        from rheojax.gui.pages.home_page import HomePage

        source = inspect.getsource(HomePage)
        assert "font-size: 48px" not in source, (
            "Hardcoded 48px font-size found — should use Typography.SIZE_HERO"
        )
        assert "font-size: 18px" not in source, (
            "Hardcoded 18px font-size found — should use Typography token"
        )
