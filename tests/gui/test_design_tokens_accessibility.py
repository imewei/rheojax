"""Tests for design token migration and theme synchronization."""

import re

import pytest

pytestmark = pytest.mark.gui


# =============================================================================
# Design Token Migration Tests
# =============================================================================


class TestDesignTokenUsage:
    """Verify pages reference design tokens instead of hardcoded values."""

    @pytest.mark.smoke
    def test_pyqtgraph_canvas_uses_theme_tokens(self) -> None:
        """PyQtGraphCanvas should use theme tokens for background and colors, not hardcoded strings."""
        import inspect

        from rheojax.gui.widgets.pyqtgraph_canvas import PyQtGraphCanvas

        source = inspect.getsource(PyQtGraphCanvas)

        # Verify no hardcoded hex strings remain for colors
        assert 'background="w"' not in source, (
            'Hardcoded background="w" found — should use themed("BG_BASE")'
        )
        assert 'foreground="k"' not in source, (
            'Hardcoded foreground="k" found — should use themed("TEXT_PRIMARY")'
        )
        assert '"#1f77b4"' not in source, (
            'Hardcoded color "#1f77b4" found — should use themed("CHART_1")'
        )
        assert "themed(" in source, (
            "PyQtGraphCanvas must use themed() for dynamic colors"
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
