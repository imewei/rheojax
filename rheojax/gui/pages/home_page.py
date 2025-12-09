"""
Home Page
=========

Landing page with quick start actions and recent projects.
"""

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
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

from rheojax.gui.state.store import StateStore
from rheojax.gui.widgets.jax_status import JAXStatusWidget


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

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize home page.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
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

    def _create_header(self) -> QWidget:
        """Create header with logo and title."""
        header = QWidget()
        layout = QVBoxLayout(header)
        layout.setSpacing(10)

        # Title
        title = QLabel("RheoJAX")
        title_font = QFont()
        title_font.setPointSize(32)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)

        # Subtitle
        subtitle = QLabel("JAX-Accelerated Rheological Analysis")
        subtitle_font = QFont()
        subtitle_font.setPointSize(14)
        subtitle.setFont(subtitle_font)
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #666;")

        # Version
        try:
            from rheojax import __version__
            version_text = f"Version {__version__}"
        except ImportError:
            version_text = "Version 0.6.0"

        version = QLabel(version_text)
        version.setAlignment(Qt.AlignCenter)
        version.setStyleSheet("color: #999; font-size: 11pt;")

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(version)

        return header

    def _create_quick_start(self) -> QWidget:
        """Create quick start section."""
        group = QGroupBox("Quick Start")
        group.setStyleSheet("""
            QGroupBox {
                font-size: 14pt;
                font-weight: bold;
                border: 2px solid #ddd;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)

        layout = QHBoxLayout(group)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 25, 15, 15)

        # Open Project button
        btn_open = self._create_action_button(
            "Open Project",
            "Load an existing RheoJAX project",
            "#2196F3"
        )
        btn_open.clicked.connect(self.open_project_requested.emit)
        layout.addWidget(btn_open)

        # Import Data button
        btn_import = self._create_action_button(
            "Import Data",
            "Import rheological data from file",
            "#4CAF50"
        )
        btn_import.clicked.connect(self.import_data_requested.emit)
        layout.addWidget(btn_import)

        # New Project button
        btn_new = self._create_action_button(
            "New Project",
            "Start a new analysis project",
            "#FF9800"
        )
        btn_new.clicked.connect(self.new_project_requested.emit)
        layout.addWidget(btn_new)

        return group

    def _create_action_button(self, title: str, description: str, color: str) -> QPushButton:
        """Create a styled action button."""
        btn = QPushButton(f"{title}\n\n{description}")
        btn.setMinimumHeight(100)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 15px;
                font-size: 12pt;
                font-weight: bold;
                text-align: center;
            }}
            QPushButton:hover {{
                background-color: {self._darken_color(color)};
            }}
            QPushButton:pressed {{
                background-color: {self._darken_color(color, 0.3)};
            }}
        """)
        return btn

    def _darken_color(self, hex_color: str, factor: float = 0.1) -> str:
        """Darken a hex color."""
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        r = int(r * (1 - factor))
        g = int(g * (1 - factor))
        b = int(b * (1 - factor))
        return f"#{r:02x}{g:02x}{b:02x}"

    def _create_recent_projects(self) -> QWidget:
        """Create recent projects section."""
        group = QGroupBox("Recent Projects")
        group.setStyleSheet("""
            QGroupBox {
                font-size: 12pt;
                font-weight: bold;
                border: 2px solid #ddd;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)

        layout = QVBoxLayout(group)
        layout.setContentsMargins(15, 25, 15, 15)
        layout.setSpacing(10)

        # Get recent projects from state
        recent_projects = self._store.get_state().recent_projects

        if not recent_projects:
            no_projects = QLabel("No recent projects")
            no_projects.setStyleSheet("color: #999; font-style: italic; font-weight: normal;")
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
        widget.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                border-radius: 5px;
                padding: 10px;
            }
            QWidget:hover {
                background-color: #e8e8e8;
            }
        """)

        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)

        # Project name
        name = QLabel(project_path.stem)
        name.setStyleSheet("font-weight: bold; font-size: 11pt;")
        layout.addWidget(name)

        # Project path
        path_label = QLabel(str(project_path))
        path_label.setStyleSheet("color: #666; font-size: 9pt; font-weight: normal;")
        path_label.setWordWrap(True)
        layout.addWidget(path_label)

        # Make clickable
        widget.mousePressEvent = lambda event: self.recent_project_opened.emit(project_path)
        widget.setCursor(Qt.PointingHandCursor)

        return widget

    def _create_examples(self) -> QWidget:
        """Create example datasets section."""
        group = QGroupBox("Example Datasets")
        group.setStyleSheet("""
            QGroupBox {
                font-size: 12pt;
                font-weight: bold;
                border: 2px solid #ddd;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)

        layout = QGridLayout(group)
        layout.setContentsMargins(15, 25, 15, 15)
        layout.setSpacing(10)

        examples = [
            ("Oscillation", "Frequency sweep data for SAOS analysis", "#3F51B5"),
            ("Relaxation", "Stress relaxation modulus data", "#009688"),
            ("Creep", "Creep compliance data", "#FF5722"),
            ("Flow", "Steady shear flow curve data", "#E91E63"),
            ("SGR", "Soft Glassy Rheology example", "#9C27B0"),
            ("SPP", "Sequence of Physical Processes (LAOS)", "#00BCD4"),
            ("TTS", "Time-Temperature Superposition data", "#FFC107"),
            ("Bayesian", "Dataset for Bayesian inference demo", "#607D8B"),
        ]

        for i, (name, description, color) in enumerate(examples):
            row = i // 2
            col = i % 2
            card = self._create_example_card(name, description, color)
            layout.addWidget(card, row, col)

        return group

    def _create_example_card(self, name: str, description: str, color: str) -> QWidget:
        """Create an example dataset card."""
        card = QWidget()
        card.setStyleSheet(f"""
            QWidget {{
                background-color: {color};
                border-radius: 8px;
                padding: 15px;
            }}
            QWidget:hover {{
                background-color: {self._darken_color(color)};
            }}
        """)
        card.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(card)
        layout.setSpacing(5)

        # Name
        name_label = QLabel(name)
        name_label.setStyleSheet("color: white; font-size: 12pt; font-weight: bold;")
        layout.addWidget(name_label)

        # Description
        desc_label = QLabel(description)
        desc_label.setStyleSheet("color: white; font-size: 9pt; font-weight: normal;")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        # Make clickable
        card.mousePressEvent = lambda event, n=name: self.example_selected.emit(n)

        return card

    def _create_system_status(self) -> QWidget:
        """Create system status section."""
        group = QGroupBox("System Status")
        group.setStyleSheet("""
            QGroupBox {
                font-size: 12pt;
                font-weight: bold;
                border: 2px solid #ddd;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)

        layout = QVBoxLayout(group)
        layout.setContentsMargins(15, 25, 15, 15)

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
            # Fallback if JAX info unavailable
            self._jax_status.update_device_list(["cpu"])
            self._jax_status.set_current_device("cpu")
            self._jax_status.set_float64_enabled(False)

    def _create_resources(self) -> QWidget:
        """Create resources section."""
        group = QGroupBox("Resources")
        group.setStyleSheet("""
            QGroupBox {
                font-size: 12pt;
                font-weight: bold;
                border: 2px solid #ddd;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)

        layout = QHBoxLayout(group)
        layout.setContentsMargins(15, 25, 15, 15)
        layout.setSpacing(15)

        resources = [
            ("Documentation", "https://rheojax.readthedocs.io"),
            ("Examples", "https://github.com/RheoJAX/rheojax/tree/main/examples"),
            ("Tutorials", "https://github.com/RheoJAX/rheojax/tree/main/notebooks"),
            ("Report Issues", "https://github.com/RheoJAX/rheojax/issues"),
        ]

        for title, url in resources:
            btn = QPushButton(title)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #f5f5f5;
                    border: 2px solid #ddd;
                    border-radius: 5px;
                    padding: 10px;
                    font-size: 10pt;
                }
                QPushButton:hover {
                    background-color: #e8e8e8;
                    border-color: #2196F3;
                }
            """)
            btn.clicked.connect(lambda checked, u=url: self._open_url(u))
            layout.addWidget(btn)

        return group

    def _open_url(self, url: str) -> None:
        """Open URL in default browser."""
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl(url))

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
        self.example_selected.emit(example_name)
