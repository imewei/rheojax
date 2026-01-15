"""
Home Page
=========

Landing page with quick start actions and recent projects.
"""

from pathlib import Path

from rheojax.gui.compat import (
    Qt,
    Signal,
    QFont,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from rheojax.gui.resources.styles import ColorPalette, Spacing
from rheojax.gui.state.store import StateStore
from rheojax.gui.widgets.jax_status import JAXStatusWidget
from rheojax.logging import get_logger

logger = get_logger(__name__)


class HomePage(QWidget):
    """Home page with getting started content.

    Features:
        - Quick start wizard
        - Recent projects list
        - Example datasets
        - Tutorial links
        - System status

    Signals
    -------
    open_project_requested : Signal()
        Emitted when user clicks Open Project
    import_data_requested : Signal()
        Emitted when user clicks Import Data
    new_project_requested : Signal()
        Emitted when user clicks New Project
    example_selected : Signal(str)
        Emitted when user selects an example
    recent_project_opened : Signal(Path)
        Emitted when user opens a recent project

    Example
    -------
    >>> page = HomePage()  # doctest: +SKIP
    >>> page.open_project_requested.connect(on_open)  # doctest: +SKIP
    """

    open_project_requested = Signal()
    import_data_requested = Signal()
    new_project_requested = Signal()
    example_selected = Signal(str)
    recent_project_opened = Signal(Path)
    workflow_selected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize home page.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        logger.debug("Initializing", class_name=self.__class__.__name__)
        self._store = StateStore()
        self.setup_ui()

    def setup_ui(self) -> None:
        """Setup user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Create scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(20)

        # Header section
        content_layout.addWidget(self._create_header())

        # Quick Start section
        content_layout.addWidget(self._create_quick_start())

        # Workflow Selector
        content_layout.addWidget(self._create_workflow_selector())

        # Two-column layout for Recent Projects and Examples
        two_col_layout = QHBoxLayout()
        two_col_layout.setSpacing(20)

        # Recent Projects (left)
        two_col_layout.addWidget(self._create_recent_projects())

        # Example Datasets (right)
        two_col_layout.addWidget(self._create_examples())

        content_layout.addLayout(two_col_layout)

        # System Status section
        content_layout.addWidget(self._create_system_status())

        # Resources section
        content_layout.addWidget(self._create_resources())

        content_layout.addStretch()

        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)

    def _create_workflow_selector(self) -> QWidget:
        """Create workflow selection section."""
        group = QGroupBox("Select Workflow")
        # Use QSS styling from light.qss - no inline style needed

        layout = QHBoxLayout(group)
        layout.setSpacing(Spacing.LG)
        layout.setContentsMargins(
            Spacing.LG, Spacing.XL + Spacing.SM, Spacing.LG, Spacing.LG
        )

        # Fitting Workflow Card
        fit_card = self._create_workflow_card(
            "Fitting Workflow",
            "Data -> Fit -> Bayesian -> Diagnostics -> Export",
            "primary",
            "fitting",
        )
        layout.addWidget(fit_card)

        # Transform Workflow Card
        transform_card = self._create_workflow_card(
            "Transform Workflow",
            "Data -> Transforms (FFT, TTS, SPP) -> Export",
            "success",
            "transform",
        )
        layout.addWidget(transform_card)

        return group

    def _create_workflow_card(
        self, title: str, description: str, variant: str, mode: str
    ) -> QPushButton:
        """Create a workflow selection card."""
        btn = QPushButton(f"{title}\n\n{description}")
        btn.setMinimumHeight(120)
        btn.setProperty("variant", variant)
        # Use design tokens for consistent styling
        if variant == "primary":
            bg_color = ColorPalette.PRIMARY
            bg_hover = ColorPalette.PRIMARY_HOVER
        else:
            bg_color = ColorPalette.SUCCESS
            bg_hover = ColorPalette.SUCCESS_HOVER

        btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {bg_color};
                color: {ColorPalette.TEXT_INVERSE};
                border: none;
                border-radius: 8px;
                text-align: left;
                padding: {Spacing.LG}px;
                font-size: 12pt;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {bg_hover};
            }}
        """
        )
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(lambda: self._select_workflow(mode))
        return btn

    def _select_workflow(self, mode: str) -> None:
        """Handle workflow selection."""
        logger.debug(
            "Quick action triggered",
            action="select_workflow",
            page="HomePage",
            workflow_mode=mode,
        )
        self._store.dispatch("SET_WORKFLOW_MODE", {"mode": mode})
        self.workflow_selected.emit(mode)
        # Navigate to data tab to start work
        logger.debug("Navigation action", target="data", page="HomePage")
        self._store.dispatch("NAVIGATE_TAB", {"tab": "data"})

    def _create_header(self) -> QWidget:
        """Create header with logo and title."""
        header = QWidget()
        layout = QVBoxLayout(header)
        layout.setSpacing(Spacing.SM)

        # Title
        title = QLabel("RheoJAX")
        title_font = QFont()
        title_font.setPointSize(32)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color: {ColorPalette.TEXT_PRIMARY};")

        # Subtitle
        subtitle = QLabel("JAX-Accelerated Rheological Analysis")
        subtitle_font = QFont()
        subtitle_font.setPointSize(14)
        subtitle.setFont(subtitle_font)
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet(f"color: {ColorPalette.TEXT_SECONDARY};")

        # Version
        try:
            from rheojax import __version__

            version_text = f"Version {__version__}"
        except ImportError:
            version_text = "Version 0.6.0"

        version = QLabel(version_text)
        version.setAlignment(Qt.AlignCenter)
        version.setStyleSheet(f"color: {ColorPalette.TEXT_MUTED}; font-size: 11pt;")

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(version)

        return header

    def _create_quick_start(self) -> QWidget:
        """Create quick start section."""
        group = QGroupBox("Quick Start")
        # QSS styling from light.qss handles groupbox appearance

        layout = QHBoxLayout(group)
        layout.setSpacing(Spacing.LG)
        layout.setContentsMargins(
            Spacing.LG, Spacing.XL + Spacing.SM, Spacing.LG, Spacing.LG
        )

        # Open Project button
        btn_open = self._create_action_button(
            "Open Project", "Load an existing RheoJAX project", "primary"
        )
        btn_open.clicked.connect(self._on_open_project_clicked)
        layout.addWidget(btn_open)

        # Import Data button
        btn_import = self._create_action_button(
            "Import Data", "Import TRIOS, Anton Paar, CSV, Excel", "success"
        )
        btn_import.clicked.connect(self._on_import_data_clicked)
        layout.addWidget(btn_import)

        # New Project button
        btn_new = self._create_action_button(
            "New Project", "Start a new analysis project", "warning"
        )
        btn_new.clicked.connect(self._on_new_project_clicked)
        layout.addWidget(btn_new)

        return group

    def _on_open_project_clicked(self) -> None:
        """Handle Open Project button click."""
        logger.debug("Quick action triggered", action="open_project", page="HomePage")
        self.open_project_requested.emit()

    def _on_import_data_clicked(self) -> None:
        """Handle Import Data button click."""
        logger.debug("Quick action triggered", action="import_data", page="HomePage")
        self.import_data_requested.emit()

    def _on_new_project_clicked(self) -> None:
        """Handle New Project button click."""
        logger.debug("Quick action triggered", action="new_project", page="HomePage")
        self.new_project_requested.emit()

    def _create_action_button(
        self, title: str, description: str, variant: str
    ) -> QPushButton:
        """Create a styled action button."""
        btn = QPushButton(f"{title}\n\n{description}")
        btn.setMinimumHeight(110)
        btn.setProperty("variant", variant)
        btn.setCursor(Qt.PointingHandCursor)

        # Use design tokens for consistent styling
        variant_colors = {
            "primary": (ColorPalette.PRIMARY, ColorPalette.PRIMARY_HOVER),
            "success": (ColorPalette.SUCCESS, ColorPalette.SUCCESS_HOVER),
            "warning": (ColorPalette.WARNING, ColorPalette.WARNING_HOVER),
            "error": (ColorPalette.ERROR, ColorPalette.ERROR_HOVER),
            "accent": (ColorPalette.ACCENT, ColorPalette.ACCENT_HOVER),
        }
        bg_color, bg_hover = variant_colors.get(
            variant, (ColorPalette.PRIMARY, ColorPalette.PRIMARY_HOVER)
        )
        text_color = (
            ColorPalette.TEXT_PRIMARY
            if variant == "warning"
            else ColorPalette.TEXT_INVERSE
        )

        btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {bg_color};
                color: {text_color};
                border: none;
                border-radius: 8px;
                text-align: left;
                padding: {Spacing.LG}px;
                font-size: 11pt;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {bg_hover};
            }}
        """
        )
        return btn

    def _darken_color(self, hex_color: str, factor: float = 0.1) -> str:
        """Darken a hex color."""
        hex_color = hex_color.lstrip("#")
        r, g, b = (
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
        )
        r = int(r * (1 - factor))
        g = int(g * (1 - factor))
        b = int(b * (1 - factor))
        return f"#{r:02x}{g:02x}{b:02x}"

    def _create_recent_projects(self) -> QWidget:
        """Create recent projects section."""
        group = QGroupBox("Recent Projects")
        # QSS styling from light.qss handles groupbox appearance

        layout = QVBoxLayout(group)
        layout.setContentsMargins(
            Spacing.LG, Spacing.XL + Spacing.SM, Spacing.LG, Spacing.LG
        )
        layout.setSpacing(Spacing.SM)

        # Get recent projects from state
        recent_projects = self._store.get_state().recent_projects

        if not recent_projects:
            no_projects = QLabel("No recent projects")
            no_projects.setProperty("class", "placeholder")
            no_projects.setAlignment(Qt.AlignCenter)
            layout.addWidget(no_projects)
        else:
            for project_path in recent_projects[:5]:  # Show last 5
                project_widget = self._create_recent_project_item(project_path)
                layout.addWidget(project_widget)

        layout.addStretch()

        return group

    def _create_recent_project_item(self, project_path: Path) -> QWidget:
        """Create a recent project item."""
        widget = QWidget()
        widget.setStyleSheet(
            f"""
            QWidget {{
                background-color: {ColorPalette.BG_SURFACE};
                border: 1px solid {ColorPalette.BORDER_DEFAULT};
                border-radius: 6px;
            }}
            QWidget:hover {{
                background-color: {ColorPalette.BG_HOVER};
                border-color: {ColorPalette.BORDER_HOVER};
            }}
        """
        )

        layout = QVBoxLayout(widget)
        layout.setContentsMargins(Spacing.MD, Spacing.MD, Spacing.MD, Spacing.MD)
        layout.setSpacing(Spacing.XS)

        # Project name
        name = QLabel(project_path.stem)
        name.setStyleSheet(
            f"font-weight: 600; font-size: 11pt; color: {ColorPalette.TEXT_PRIMARY};"
        )
        layout.addWidget(name)

        # Project path
        path_label = QLabel(str(project_path))
        path_label.setStyleSheet(
            f"color: {ColorPalette.TEXT_MUTED}; font-size: 9pt; font-weight: normal;"
        )
        path_label.setWordWrap(True)
        layout.addWidget(path_label)

        # Make clickable
        def on_project_click(event, path=project_path):
            logger.debug(
                "Quick action triggered",
                action="open_recent_project",
                page="HomePage",
                project_path=str(path),
            )
            self.recent_project_opened.emit(path)

        widget.mousePressEvent = on_project_click
        widget.setCursor(Qt.PointingHandCursor)

        return widget

    def _create_examples(self) -> QWidget:
        """Create example datasets section."""
        group = QGroupBox("Example Datasets")
        # QSS styling from light.qss handles groupbox appearance

        layout = QGridLayout(group)
        layout.setContentsMargins(
            Spacing.LG, Spacing.XL + Spacing.SM, Spacing.LG, Spacing.LG
        )
        layout.setHorizontalSpacing(Spacing.MD)
        layout.setVerticalSpacing(Spacing.MD)
        self._examples_layout = layout
        self._example_cards: list[QWidget] = []

        # Map example names to actual file paths (relative to project root)
        self._example_paths = {
            "Oscillation": "examples/data/pyRheo/chia_pudding/oscillation_chia_data.csv",
            "Relaxation": "examples/data/experimental/polypropylene_relaxation.csv",
            "Creep": "examples/data/pyRheo/mucus/creep_mucus_data.csv",
            "Flow": "examples/data/experimental/cellulose_hydrogel_flow.csv",
            "SGR": "examples/data/pyRheo/emulsion/emulsions_v2.csv",
            "SPP": "examples/data/experimental/multi_technique.txt",
            "TTS": "examples/data/experimental/frequency_sweep_tts.txt",
            "Bayesian": "examples/data/pyRheo/fish_muscle/stressrelaxation_fishmuscle_data.csv",
        }

        # Use design tokens for chart colors
        chart_colors = [
            ColorPalette.CHART_1,  # Blue
            ColorPalette.CHART_3,  # Green
            ColorPalette.CHART_5,  # Red
            ColorPalette.CHART_6,  # Pink
            ColorPalette.CHART_2,  # Purple
            ColorPalette.INFO,  # Cyan-ish
            ColorPalette.CHART_4,  # Amber
            ColorPalette.TEXT_MUTED,  # Gray
        ]

        examples = [
            ("Oscillation", "SAOS frequency sweep (CSV)"),
            ("Relaxation", "Stress relaxation G(t) (CSV)"),
            ("Creep", "Creep compliance J(t) (CSV)"),
            ("Flow", "Steady shear flow curve (CSV)"),
            ("SGR", "Soft glassy emulsion (CSV)"),
            ("SPP", "Multi-technique TRIOS (TXT)"),
            ("TTS", "Time-Temperature Superposition (TXT)"),
            ("Bayesian", "Fish muscle relaxation (CSV)"),
        ]

        for idx, (name, description) in enumerate(examples):
            color = chart_colors[idx % len(chart_colors)]
            card = self._create_example_card(name, description, color)
            self._example_cards.append(card)

        self._relayout_examples()

        return group

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._relayout_examples()

    def _relayout_examples(self) -> None:
        if not hasattr(self, "_examples_layout"):
            return
        layout = self._examples_layout
        # Clear layout
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        if not self._example_cards:
            return

        # Determine columns based on available width
        width = self.width()
        cols = 3 if width > 1100 else 2 if width > 720 else 1

        for idx, card in enumerate(self._example_cards):
            row = idx // cols
            col = idx % cols
            layout.addWidget(card, row, col)

    def _create_example_card(self, name: str, description: str, color: str) -> QWidget:
        """Create an example dataset card."""
        card = QWidget()
        card.setStyleSheet(
            f"""
            QWidget {{
                background-color: {color};
                border-radius: 8px;
            }}
            QWidget:hover {{
                background-color: {self._darken_color(color)};
            }}
        """
        )
        card.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(Spacing.LG, Spacing.LG, Spacing.LG, Spacing.LG)
        layout.setSpacing(Spacing.XS)

        # Name
        name_label = QLabel(name)
        name_label.setStyleSheet(
            f"color: {ColorPalette.TEXT_INVERSE}; font-size: 12pt; font-weight: 600;"
        )
        layout.addWidget(name_label)

        # Description
        desc_label = QLabel(description)
        desc_label.setStyleSheet(
            f"color: {ColorPalette.TEXT_INVERSE}; font-size: 9pt; font-weight: normal;"
        )
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        # Make clickable
        def on_example_click(event, example_name=name):
            logger.debug(
                "Quick action triggered",
                action="select_example",
                page="HomePage",
                example_name=example_name,
            )
            self.example_selected.emit(example_name)

        card.mousePressEvent = on_example_click

        return card

    def _create_system_status(self) -> QWidget:
        """Create system status section."""
        group = QGroupBox("System Status")
        # QSS styling from light.qss handles groupbox appearance

        layout = QVBoxLayout(group)
        layout.setContentsMargins(
            Spacing.LG, Spacing.XL + Spacing.SM, Spacing.LG, Spacing.LG
        )

        # JAX status widget
        self._jax_status = JAXStatusWidget()
        layout.addWidget(self._jax_status)

        # Initialize JAX status
        self._update_jax_status()

        return group

    def _update_jax_status(self) -> None:
        """Update JAX status display."""
        try:
            from rheojax.gui.utils.jax_utils import get_jax_info

            jax_info = get_jax_info()

            # Update device list
            devices = jax_info.get("devices", ["cpu"])
            self._jax_status.update_device_list(devices)

            # Set current device
            current_device = jax_info.get("default_device", "cpu")
            self._jax_status.set_current_device(current_device)

            # Update memory
            memory_used = jax_info.get("memory_used_mb", 0)
            memory_total = jax_info.get("memory_total_mb", 0)
            self._jax_status.update_memory(memory_used, memory_total)

            # Float64 status
            float64_enabled = jax_info.get("float64_enabled", False)
            self._jax_status.set_float64_enabled(float64_enabled)

            # JIT cache
            jit_cache_count = jax_info.get("jit_cache_count", 0)
            self._jax_status.update_jit_cache(jit_cache_count)

        except Exception:
            logger.error("Failed to update JAX status", exc_info=True, page="HomePage")
            # Fallback if JAX info unavailable
            self._jax_status.update_device_list(["cpu"])
            self._jax_status.set_current_device("cpu")
            self._jax_status.set_float64_enabled(False)

    def _create_resources(self) -> QWidget:
        """Create resources section."""
        group = QGroupBox("Resources")
        # QSS styling from light.qss handles groupbox appearance

        layout = QHBoxLayout(group)
        layout.setContentsMargins(
            Spacing.LG, Spacing.XL + Spacing.SM, Spacing.LG, Spacing.LG
        )
        layout.setSpacing(Spacing.LG)

        resources = [
            ("Documentation", "https://rheojax.readthedocs.io"),
            ("Tutorials", "https://github.com/imewei/rheojax/tree/main/examples"),
            ("Report Issues", "https://github.com/imewei/rheojax/issues"),
        ]

        for title, url in resources:
            btn = QPushButton(title)
            btn.setProperty("variant", "secondary")
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: {ColorPalette.BG_SURFACE};
                    color: {ColorPalette.TEXT_PRIMARY};
                    border: 1px solid {ColorPalette.BORDER_DEFAULT};
                    border-radius: 6px;
                    padding: {Spacing.MD}px {Spacing.LG}px;
                    font-size: 11pt;
                    font-weight: 500;
                }}
                QPushButton:hover {{
                    background-color: {ColorPalette.PRIMARY_LIGHT};
                    border-color: {ColorPalette.PRIMARY};
                    color: {ColorPalette.PRIMARY};
                }}
                QPushButton:pressed {{
                    background-color: {ColorPalette.BG_ACTIVE};
                }}
            """
            )
            btn.clicked.connect(lambda checked, u=url, t=title: self._open_url(u, t))
            layout.addWidget(btn)

        return group

    def _open_url(self, url: str, title: str = "") -> None:
        """Open URL in default browser."""
        from rheojax.gui.compat import QtCore, QtGui

        logger.debug("Navigation action", target=url, page="HomePage", resource=title)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))

    def load_recent_projects(self) -> list[dict[str, str]]:
        """Load recent project list.

        Returns
        -------
        list[dict]
            Recent projects with metadata
        """
        state = self._store.get_state()
        return [
            {
                "path": str(path),
                "name": path.stem,
                "modified": path.stat().st_mtime if path.exists() else 0,
            }
            for path in state.recent_projects
        ]

    def open_example(self, example_name: str) -> None:
        """Open example dataset.

        Parameters
        ----------
        example_name : str
            Example identifier
        """
        logger.debug(
            "Quick action triggered",
            action="open_example",
            page="HomePage",
            example_name=example_name,
        )
        self.example_selected.emit(example_name)

    def get_example_path(self, example_name: str) -> Path | None:
        """Get the file path for an example dataset.

        Parameters
        ----------
        example_name : str
            Example identifier (e.g., "Oscillation", "Relaxation")

        Returns
        -------
        Path | None
            Absolute path to the example file, or None if not found
        """
        if not hasattr(self, "_example_paths"):
            return None

        rel_path = self._example_paths.get(example_name)
        if not rel_path:
            return None

        # Try resolving from project root (current working directory)
        path = Path(rel_path)
        if path.exists():
            return path.resolve()

        # Try from cwd
        cwd_path = Path.cwd() / rel_path
        if cwd_path.exists():
            return cwd_path.resolve()

        # Try from module location
        module_dir = Path(__file__).parent.parent.parent.parent
        module_path = module_dir / rel_path
        if module_path.exists():
            return module_path.resolve()

        return None
