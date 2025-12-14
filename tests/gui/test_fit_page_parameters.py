import pytest


@pytest.mark.gui
def test_fit_page_restores_parameter_section(qapp) -> None:
    from PySide6.QtWidgets import QGroupBox

    from rheojax.gui.pages.fit_page import FitPage

    page = FitPage()
    try:
        group_titles = {g.title() for g in page.findChildren(QGroupBox)}
        assert "Parameters" in group_titles
        assert hasattr(page, "_parameter_table")
    finally:
        page.deleteLater()


@pytest.mark.gui
def test_fit_page_invalid_model_does_not_crash(qapp) -> None:
    from rheojax.gui.pages.fit_page import FitPage

    page = FitPage()
    try:
        page._apply_model_selection("not_a_real_model", dispatch=False)
        assert page._parameter_table.isEnabled() is False
        assert page._btn_fit.isEnabled() is False
    finally:
        page.deleteLater()
