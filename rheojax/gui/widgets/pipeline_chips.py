"""
Pipeline Chips Widget
====================

Visual pipeline representation with status indicators.
"""

from rheojax.gui.compat import (
    QColor,
    QFont,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    Qt,
    QTimer,
    QWidget,
    Signal,
)
from rheojax.gui.state.store import PipelineStep, StepStatus
from rheojax.logging import get_logger

logger = get_logger(__name__)


class PipelineChips(QWidget):
    """Horizontal pipeline progress widget with status indicators.

    Features:
        - 5 chips: Load, Transform, Fit, Bayesian, Export
        - Status indicators (pending, active, complete, warning, error)
        - Clickable chips to navigate to corresponding tab
        - Visual arrows/connectors between chips
        - Active chip shows spinner animation

    Signals
    -------
    step_clicked : Signal(PipelineStep)
        Emitted when a chip is clicked

    Example
    -------
    >>> chips = PipelineChips()  # doctest: +SKIP
    >>> chips.set_step_status(PipelineStep.LOAD, StepStatus.COMPLETE)  # doctest: +SKIP
    >>> chips.step_clicked.connect(on_step_clicked)  # doctest: +SKIP
    """

    step_clicked = Signal(PipelineStep)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize pipeline chips.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        logger.debug("Initializing", class_name=self.__class__.__name__)

        # Main horizontal layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(5)

        # Create chips for each pipeline step
        self._chips: dict[PipelineStep, QPushButton] = {}
        self._arrows: list[QLabel] = []

        steps = [
            (PipelineStep.LOAD, "Load"),
            (PipelineStep.TRANSFORM, "Transform"),
            (PipelineStep.FIT, "Fit"),
            (PipelineStep.BAYESIAN, "Bayesian"),
            (PipelineStep.EXPORT, "Export"),
        ]

        for i, (step, label) in enumerate(steps):
            # Create chip button
            chip = self._create_chip(step, label)
            self._chips[step] = chip
            layout.addWidget(chip)

            # Add arrow connector between chips (except after last)
            if i < len(steps) - 1:
                arrow = self._create_arrow()
                self._arrows.append(arrow)
                layout.addWidget(arrow)

        layout.addStretch()

        # Timer for spinner animation on active chips
        self._spinner_timer = QTimer(self)
        self._spinner_timer.timeout.connect(self._update_spinner)
        self._spinner_timer.start(500)  # Update every 500ms
        self._spinner_state = 0
        logger.debug("Initialization complete", class_name=self.__class__.__name__)

    def _create_chip(self, step: PipelineStep, label: str) -> QPushButton:
        """Create a chip button for a pipeline step.

        Parameters
        ----------
        step : PipelineStep
            Pipeline step
        label : str
            Button label

        Returns
        -------
        QPushButton
            Chip button
        """
        chip = QPushButton(label)
        chip.setMinimumWidth(100)
        chip.setMaximumWidth(120)
        chip.setMinimumHeight(35)
        chip.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # Make clickable
        chip.clicked.connect(lambda: self._on_chip_clicked(step))

        # Set initial status (pending)
        self._apply_chip_style(chip, StepStatus.PENDING)

        return chip

    def _on_chip_clicked(self, step: PipelineStep) -> None:
        """Handle chip click.

        Parameters
        ----------
        step : PipelineStep
            Clicked pipeline step
        """
        logger.debug(
            "User interaction",
            widget=self.__class__.__name__,
            action="chip_clicked",
            step=step.name if hasattr(step, "name") else str(step),
        )
        self.step_clicked.emit(step)

    def _create_arrow(self) -> QLabel:
        """Create an arrow connector between chips.

        Returns
        -------
        QLabel
            Arrow label
        """
        arrow = QLabel("→")
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        arrow.setFont(font)
        arrow.setAlignment(Qt.AlignCenter)
        arrow.setStyleSheet("color: #cccccc;")
        return arrow

    def set_step_status(self, step: PipelineStep, status: StepStatus) -> None:
        """Set the status of a pipeline step.

        Parameters
        ----------
        step : PipelineStep
            Pipeline step
        status : StepStatus
            New status
        """
        if step not in self._chips:
            return

        logger.debug(
            "State updated",
            widget=self.__class__.__name__,
            action="set_step_status",
            step=step.name if hasattr(step, "name") else str(step),
            status=status.name if hasattr(status, "name") else str(status),
        )

        chip = self._chips[step]
        self._apply_chip_style(chip, status)

    def reset_all(self) -> None:
        """Reset all chips to pending status."""
        logger.debug(
            "State updated",
            widget=self.__class__.__name__,
            action="reset_all",
        )
        for chip in self._chips.values():
            self._apply_chip_style(chip, StepStatus.PENDING)

    def _apply_chip_style(self, chip: QPushButton, status: StepStatus) -> None:
        """Apply styling to chip based on status.

        Parameters
        ----------
        chip : QPushButton
            Chip button
        status : StepStatus
            Status to apply
        """
        # Status-based colors
        if status == StepStatus.PENDING:
            bg_color = "#e0e0e0"
            text_color = "#666666"
            border_color = "#cccccc"
        elif status == StepStatus.ACTIVE:
            bg_color = "#2196F3"
            text_color = "#ffffff"
            border_color = "#1976D2"
        elif status == StepStatus.COMPLETE:
            bg_color = "#4CAF50"
            text_color = "#ffffff"
            border_color = "#388E3C"
        elif status == StepStatus.WARNING:
            bg_color = "#FF9800"
            text_color = "#ffffff"
            border_color = "#F57C00"
        elif status == StepStatus.ERROR:
            bg_color = "#F44336"
            text_color = "#ffffff"
            border_color = "#D32F2F"
        else:
            bg_color = "#e0e0e0"
            text_color = "#666666"
            border_color = "#cccccc"

        # Apply stylesheet
        chip.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {bg_color};
                color: {text_color};
                border: 2px solid {border_color};
                border-radius: 18px;
                padding: 5px 15px;
                font-weight: bold;
                font-size: 11pt;
            }}
            QPushButton:hover {{
                background-color: {self._lighten_color(bg_color)};
            }}
            QPushButton:pressed {{
                background-color: {self._darken_color(bg_color)};
            }}
        """
        )

        # Store status for spinner animation
        chip.setProperty("status", status)

    def _lighten_color(self, hex_color: str) -> str:
        """Lighten a hex color by 10%.

        Parameters
        ----------
        hex_color : str
            Hex color string

        Returns
        -------
        str
            Lightened hex color
        """
        color = QColor(hex_color)
        hue, sat, light, alpha = color.getHsl()
        light = min(255, int(light * 1.1))
        color.setHsl(hue, sat, light, alpha)
        return color.name()

    def _darken_color(self, hex_color: str) -> str:
        """Darken a hex color by 10%.

        Parameters
        ----------
        hex_color : str
            Hex color string

        Returns
        -------
        str
            Darkened hex color
        """
        color = QColor(hex_color)
        hue, sat, light, alpha = color.getHsl()
        light = max(0, int(light * 0.9))
        color.setHsl(hue, sat, light, alpha)
        return color.name()

    def _update_spinner(self) -> None:
        """Update spinner animation for active chips."""
        self._spinner_state = (self._spinner_state + 1) % 4

        spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

        for _step, chip in self._chips.items():
            status = chip.property("status")
            if status == StepStatus.ACTIVE:
                # Get base text (remove previous spinner if present)
                base_text = chip.text()
                for char in spinner_chars:
                    base_text = base_text.replace(f" {char}", "")

                # Add spinner
                spinner_char = spinner_chars[self._spinner_state % len(spinner_chars)]
                chip.setText(f"{base_text} {spinner_char}")
            elif status == StepStatus.COMPLETE:
                # Show a checkmark
                base_text = chip.text()
                for char in spinner_chars:
                    base_text = base_text.replace(f" {char}", "")
                chip.setText(base_text.replace(" ✔", "") + " ✔")
            else:
                # Ensure spinner not shown for other states
                base_text = chip.text()
                for char in spinner_chars:
                    base_text = base_text.replace(f" {char}", "")
                chip.setText(base_text)
