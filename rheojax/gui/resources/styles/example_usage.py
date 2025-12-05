"""
Example usage of RheoJAX GUI stylesheets.

This demonstrates how to apply themes to a PyQt6 application.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from rheojax.gui.resources.styles import get_stylesheet


class ThemeDemo(QMainWindow):
    """Demo window showcasing styled widgets."""

    def __init__(self):
        super().__init__()
        self.current_theme = "light"
        self._init_ui()
        self._apply_theme()

    def _init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("RheoJAX Theme Demo")
        self.setGeometry(100, 100, 800, 600)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Theme toggle button
        theme_btn = QPushButton("Toggle Theme")
        theme_btn.clicked.connect(self.toggle_theme)
        layout.addWidget(theme_btn)

        # Tab widget
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Tab 1: Buttons
        buttons_tab = self._create_buttons_tab()
        tabs.addTab(buttons_tab, "Buttons")

        # Tab 2: Inputs
        inputs_tab = self._create_inputs_tab()
        tabs.addTab(inputs_tab, "Inputs")

        # Tab 3: Progress & Feedback
        progress_tab = self._create_progress_tab()
        tabs.addTab(progress_tab, "Progress")

    def _create_buttons_tab(self) -> QWidget:
        """Create buttons demonstration tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Button variants
        group = QGroupBox("Button Variants")
        group_layout = QVBoxLayout(group)

        # Primary button
        btn_primary = QPushButton("Primary Button")
        group_layout.addWidget(btn_primary)

        # Accent button (purple - for Bayesian)
        btn_accent = QPushButton("Bayesian Analysis")
        btn_accent.setObjectName("accentButton")
        group_layout.addWidget(btn_accent)

        # Success button
        btn_success = QPushButton("Success")
        btn_success.setObjectName("successButton")
        group_layout.addWidget(btn_success)

        # Warning button
        btn_warning = QPushButton("Warning")
        btn_warning.setObjectName("warningButton")
        group_layout.addWidget(btn_warning)

        # Error button
        btn_error = QPushButton("Delete")
        btn_error.setObjectName("errorButton")
        group_layout.addWidget(btn_error)

        # Secondary button
        btn_secondary = QPushButton("Cancel")
        btn_secondary.setObjectName("secondaryButton")
        group_layout.addWidget(btn_secondary)

        # Disabled button
        btn_disabled = QPushButton("Disabled")
        btn_disabled.setEnabled(False)
        group_layout.addWidget(btn_disabled)

        layout.addWidget(group)
        layout.addStretch()

        return widget

    def _create_inputs_tab(self) -> QWidget:
        """Create inputs demonstration tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Text inputs
        text_group = QGroupBox("Text Inputs")
        text_layout = QVBoxLayout(text_group)

        text_layout.addWidget(QLabel("Line Edit:"))
        line_edit = QLineEdit()
        line_edit.setPlaceholderText("Enter text here...")
        text_layout.addWidget(line_edit)

        text_layout.addWidget(QLabel("Number Input:"))
        spin_box = QDoubleSpinBox()
        spin_box.setValue(3.14159)
        text_layout.addWidget(spin_box)

        text_layout.addWidget(QLabel("Dropdown:"))
        combo = QComboBox()
        combo.addItems(["Option 1", "Option 2", "Option 3"])
        text_layout.addWidget(combo)

        layout.addWidget(text_group)

        # Selection inputs
        select_group = QGroupBox("Selection Inputs")
        select_layout = QVBoxLayout(select_group)

        checkbox1 = QCheckBox("Enable feature A")
        checkbox1.setChecked(True)
        select_layout.addWidget(checkbox1)

        checkbox2 = QCheckBox("Enable feature B")
        select_layout.addWidget(checkbox2)

        radio1 = QRadioButton("Option A")
        radio1.setChecked(True)
        select_layout.addWidget(radio1)

        radio2 = QRadioButton("Option B")
        select_layout.addWidget(radio2)

        layout.addWidget(select_group)

        # Slider
        slider_group = QGroupBox("Slider")
        slider_layout = QVBoxLayout(slider_group)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setValue(50)
        slider_layout.addWidget(slider)
        layout.addWidget(slider_group)

        layout.addStretch()

        return widget

    def _create_progress_tab(self) -> QWidget:
        """Create progress and feedback tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Progress bars
        progress_group = QGroupBox("Progress Bars")
        progress_layout = QVBoxLayout(progress_group)

        progress_layout.addWidget(QLabel("Primary Progress:"))
        progress1 = QProgressBar()
        progress1.setValue(60)
        progress_layout.addWidget(progress1)

        progress_layout.addWidget(QLabel("Success Progress:"))
        progress2 = QProgressBar()
        progress2.setObjectName("successProgress")
        progress2.setValue(80)
        progress_layout.addWidget(progress2)

        progress_layout.addWidget(QLabel("Warning Progress:"))
        progress3 = QProgressBar()
        progress3.setObjectName("warningProgress")
        progress3.setValue(45)
        progress_layout.addWidget(progress3)

        progress_layout.addWidget(QLabel("Error Progress:"))
        progress4 = QProgressBar()
        progress4.setObjectName("errorProgress")
        progress4.setValue(25)
        progress_layout.addWidget(progress4)

        layout.addWidget(progress_group)

        # Text area
        text_group = QGroupBox("Text Area")
        text_layout = QVBoxLayout(text_group)
        text_edit = QTextEdit()
        text_edit.setPlainText(
            "This is a styled text area.\n\n"
            "It supports multiple lines and scrolling.\n"
            "The styling includes proper borders, padding, and focus states."
        )
        text_layout.addWidget(text_edit)
        layout.addWidget(text_group)

        layout.addStretch()

        return widget

    def _apply_theme(self):
        """Apply the current theme to the application."""
        stylesheet = get_stylesheet(self.current_theme)
        self.setStyleSheet(stylesheet)
        self.statusBar().showMessage(f"Theme: {self.current_theme.capitalize()}")

    def toggle_theme(self):
        """Toggle between light and dark themes."""
        self.current_theme = "dark" if self.current_theme == "light" else "light"
        self._apply_theme()


def main():
    """Run the theme demo application."""
    import sys

    app = QApplication(sys.argv)
    demo = ThemeDemo()
    demo.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
