"""
About Dialog
===========

Application information and credits.
"""

import webbrowser

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from rheojax import __version__


class AboutDialog(QDialog):
    """About dialog with version info.

    Content:
        - RheoJAX version and logo
        - Description
        - Key dependencies
        - License information
        - Links (Documentation, GitHub, Issues)
        - Credits

    Example
    -------
    >>> dialog = AboutDialog()  # doctest: +SKIP
    >>> dialog.exec()  # doctest: +SKIP
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize about dialog.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)

        self.setWindowTitle("About RheoJAX")
        self.setMinimumSize(550, 600)

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up user interface."""
        layout = QVBoxLayout()

        # Logo/Title section
        title_layout = QVBoxLayout()
        title_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # App name
        app_name = QLabel("RheoJAX")
        app_name_font = QFont()
        app_name_font.setPointSize(24)
        app_name_font.setBold(True)
        app_name.setFont(app_name_font)
        app_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_layout.addWidget(app_name)

        # Version
        version_label = QLabel(f"Version {__version__}")
        version_font = QFont()
        version_font.setPointSize(12)
        version_label.setFont(version_font)
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_layout.addWidget(version_label)

        layout.addLayout(title_layout)
        layout.addSpacing(20)

        # Description
        description = QLabel(
            "JAX-accelerated rheological analysis package with 2-10x "
            "performance improvements through JAX + GPU acceleration."
        )
        description.setWordWrap(True)
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(description)

        layout.addSpacing(10)

        # Info text browser
        info_browser = QTextBrowser()
        info_browser.setOpenExternalLinks(False)
        info_browser.anchorClicked.connect(self._on_link_clicked)

        info_html = """
<html>
<body>
<h3>Key Features</h3>
<ul>
<li><b>23 Rheological Models:</b> Classical, Fractional, Flow, Multi-Mode, SGR</li>
<li><b>Fast Optimization:</b> NLSQ with 5-270x speedup vs scipy</li>
<li><b>Bayesian Inference:</b> NumPyro NUTS with comprehensive diagnostics</li>
<li><b>Advanced Transforms:</b> Mastercurve (TTS), SRFS, FFT, derivatives</li>
<li><b>Multi-Format I/O:</b> TRIOS, CSV, Excel, HDF5, Anton Paar</li>
</ul>

<h3>Key Dependencies</h3>
<ul>
<li><b>JAX/jaxlib</b> 0.8.0 - GPU-accelerated computation</li>
<li><b>NLSQ</b> ≥0.3.0 - Fast nonlinear least squares</li>
<li><b>NumPyro</b> ≥0.19.0 - Bayesian inference with NUTS</li>
<li><b>ArviZ</b> ≥0.22.0 - Bayesian diagnostics and visualization</li>
<li><b>NumPy</b> ≥2.3.0, <b>SciPy</b> ≥1.16.0, <b>Pandas</b> ≥2.3.0</li>
<li><b>Matplotlib</b> ≥3.10.0 - Visualization</li>
<li><b>PySide6</b> ≥6.6.0 - Qt6 GUI framework</li>
</ul>

<h3>License</h3>
<p>
RheoJAX is released under the <b>MIT License</b>.<br>
Copyright (c) 2024 Wei Chen
</p>

<h3>Links</h3>
<p>
<a href="https://rheojax.readthedocs.io">Documentation</a> |
<a href="https://github.com/imewei/rheojax">GitHub Repository</a> |
<a href="https://github.com/imewei/rheojax/issues">Report Issues</a>
</p>

<h3>Credits</h3>
<p>
<b>Author:</b> Wei Chen (wchen@anl.gov)<br>
<b>Institution:</b> Argonne National Laboratory
</p>

<p style="font-size: 10px; color: #666;">
Built with Python 3.12+, JAX, NumPyro, and Qt6
</p>
</body>
</html>
        """

        info_browser.setHtml(info_html)
        layout.addWidget(info_browser)

        # Close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.accept)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def _on_link_clicked(self, url) -> None:
        """Handle link clicks.

        Parameters
        ----------
        url : QUrl
            Clicked URL
        """
        url_str = url.toString()
        webbrowser.open(url_str)


# Alias for backward compatibility
About = AboutDialog
