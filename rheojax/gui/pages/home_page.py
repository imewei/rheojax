"""
Home Page
=========

Landing page with quick start actions and recent projects.
"""

from pathlib import Path

from rheojax.gui.compat import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    Qt,
    QVBoxLayout,
    QWidget,
    Signal,
)
from rheojax.gui.resources.styles import (
    BorderRadius,
    ColorPalette,
    Spacing,
    Typography,
    button_style,
    themed,
)
from rheojax.gui.state.store import StateStore
from rheojax.gui.widgets.jax_status import JAXStatusWidget
from rheojax.logging import get_logger

logger = get_logger(__name__)


class ClickableWidget(QWidget):
    """A QWidget subclass that emits a signal on click and handles keyboard activation."""

    clicked = Signal()

    def mousePressEvent(self, event) -> None:
        self.clicked.emit()
        super().mousePressEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Return, Qt.Key_Space):
            self.clicked.emit()
        else:
            super().keyPressEvent(event)


class HomePage(QWidget):
    """Home page with getting started content.

    Features:
        - Quick start buttons (Import Data, New Project, Open Project)
        - Recent projects list
        - System status
        - Resources links

    Signals
    -------
    open_project_requested : Signal()
        Emitted when user clicks Open Project
    import_data_requested : Signal()
        Emitted when user clicks Import Data
    new_project_requested : Signal()
        Emitted when user clicks New Project
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
    recent_project_opened = Signal(Path)

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
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        content_widget = QWidget()
        content_widget.setStyleSheet(f"background-color: {themed('BG_CANVAS')};")
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(0)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # Hero header section (gradient banner)
        content_layout.addWidget(self._create_header())

        # Content area with padding
        body = QWidget()
        body.setStyleSheet(f"background-color: {themed('BG_CANVAS')};")
        body_layout = QVBoxLayout(body)
        body_layout.setSpacing(Spacing.XL)
        body_layout.setContentsMargins(
            Spacing.PAGE_MARGIN + 4,
            Spacing.XL,
            Spacing.PAGE_MARGIN + 4,
            Spacing.PAGE_MARGIN,
        )

        # Quick Start section
        body_layout.addWidget(self._create_quick_start())

        # Recent Projects
        body_layout.addWidget(self._create_recent_projects())

        # Bottom row: System Status and Resources side by side
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(Spacing.XL)
        bottom_row.addWidget(self._create_system_status())
        bottom_row.addWidget(self._create_resources())
        body_layout.addLayout(bottom_row)

        body_layout.addStretch()

        content_layout.addWidget(body)

        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)

    def _create_header(self) -> QWidget:
        """Create hero header with gradient background."""
        header = QWidget()
        header.setMinimumHeight(240)
        header.setStyleSheet(f"""
            QWidget {{
                background-color: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 {themed('PRIMARY_PRESSED')}, stop:0.4 {themed('PRIMARY_PRESSED')}, stop:1 {themed('ACCENT_PRESSED')}
                );
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            }}
            QLabel {{ background-color: transparent; }}
        """)

        layout = QVBoxLayout(header)
        layout.setSpacing(Spacing.MD)
        layout.setContentsMargins(
            Spacing.PAGE_MARGIN + 16,
            Spacing.XXL,
            Spacing.PAGE_MARGIN + 16,
            Spacing.XL,
        )

        # Top Row: Title + Version Badge
        top_row = QHBoxLayout()
        top_row.setSpacing(Spacing.LG)
        top_row.setContentsMargins(0, 0, 0, 0)

        # Title
        title = QLabel("RheoJAX")
        title.setStyleSheet(f"""
            color: {ColorPalette.TEXT_INVERSE};
            font-family: {Typography.FONT_FAMILY};
            font-size: {Typography.SIZE_HERO}pt;
            font-weight: {Typography.WEIGHT_BOLD};
            background-color: transparent;
        """)
        top_row.addWidget(title)

        # Version Badge
        try:
            from rheojax import __version__

            version_text = f"v{__version__}"
        except ImportError:
            version_text = "v0.6.0"

        version_badge = QLabel(version_text)
        version_badge.setStyleSheet(f"""
            QLabel {{
                color: {ColorPalette.TEXT_INVERSE};
                background-color: rgba(255, 255, 255, 0.15);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: {BorderRadius.MD}px;
                padding: {Spacing.XS}px {Spacing.MD}px;
                font-size: {Typography.SIZE_MD}pt;
                font-weight: {Typography.WEIGHT_SEMIBOLD};
                font-family: {Typography.FONT_FAMILY};
            }}
        """)
        top_row.addWidget(version_badge)
        top_row.addStretch()  # Push everything to left

        layout.addLayout(top_row)

        # Subtitle
        subtitle = QLabel("JAX-Accelerated Rheological Analysis")
        subtitle.setStyleSheet(f"""
            color: {ColorPalette.TEXT_DISABLED};
            font-family: {Typography.FONT_FAMILY};
            font-size: {Typography.SIZE_XL + 2}pt;
            font-weight: {Typography.WEIGHT_NORMAL};
            background-color: transparent;
        """)
        layout.addWidget(subtitle)

        layout.addStretch()

        return header

    def _create_quick_start(self) -> QWidget:
        """Create quick start section."""
        group = QGroupBox("Quick Start")

        layout = QHBoxLayout(group)
        layout.setSpacing(Spacing.MD)
        layout.setContentsMargins(
            Spacing.LG, Spacing.XL + Spacing.SM, Spacing.LG, Spacing.LG
        )

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

        # Open Project button
        btn_open = self._create_action_button(
            "Open Project", "Load an existing RheoJAX project", "primary"
        )
        btn_open.clicked.connect(self._on_open_project_clicked)
        layout.addWidget(btn_open)

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
        btn.setMinimumHeight(100)
        btn.setProperty("variant", variant)
        btn.setCursor(Qt.PointingHandCursor)

        # Gradient backgrounds for each variant (theme-aware tokens)
        variant_gradients = {
            "primary": (
                f"qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {themed('PRIMARY_HOVER')}, stop:1 {themed('PRIMARY')})",
                f"qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {themed('PRIMARY')}, stop:1 {themed('PRIMARY_PRESSED')})",
            ),
            "success": (
                f"qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {themed('SUCCESS_BRIGHT')}, stop:1 {themed('SUCCESS')})",
                f"qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {themed('SUCCESS')}, stop:1 {themed('SUCCESS_HOVER')})",
            ),
            "warning": (
                f"qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {themed('WARNING_BRIGHT')}, stop:1 {themed('WARNING')})",
                f"qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {themed('WARNING')}, stop:1 {themed('WARNING_HOVER')})",
            ),
            "error": (
                f"qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {themed('ERROR_BRIGHT')}, stop:1 {themed('ERROR')})",
                f"qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {themed('ERROR')}, stop:1 {themed('ERROR_HOVER')})",
            ),
            "accent": (
                f"qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {themed('ACCENT_BRIGHT')}, stop:1 {themed('ACCENT')})",
                f"qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {themed('ACCENT')}, stop:1 {themed('ACCENT_HOVER')})",
            ),
        }
        bg_grad, bg_hover_grad = variant_gradients.get(
            variant, variant_gradients["primary"]
        )
        text_color = themed("TEXT_INVERSE")

        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_grad};
                color: {text_color};
                border: none;
                border-radius: {BorderRadius.XL}px;
                text-align: left;
                padding: {Spacing.LG}px {Spacing.XL}px;
                font-size: 11pt;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {bg_hover_grad};
            }}
        """)
        return btn

    def _create_recent_projects(self) -> QWidget:
        """Create recent projects section."""
        group = QGroupBox("Recent Projects")

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
        widget = ClickableWidget()
        widget.setStyleSheet(f"""
            QWidget {{
                background-color: {ColorPalette.BG_SURFACE};
                border: 1px solid {ColorPalette.BORDER_SUBTLE};
                border-radius: {BorderRadius.LG}px;
            }}
            QWidget:hover {{
                background-color: {ColorPalette.BG_HOVER};
                border-color: {ColorPalette.BORDER_DEFAULT};
            }}
        """)

        layout = QVBoxLayout(widget)
        layout.setContentsMargins(Spacing.MD, Spacing.MD, Spacing.MD, Spacing.MD)
        layout.setSpacing(Spacing.XS)

        # Project name
        name = QLabel(project_path.stem)
        name.setStyleSheet(
            f"font-weight: 600; font-size: 11pt; color: {ColorPalette.TEXT_PRIMARY};"
            " border: none; background-color: transparent;"
        )
        layout.addWidget(name)

        # Project path
        path_label = QLabel(str(project_path))
        path_label.setStyleSheet(
            f"color: {ColorPalette.TEXT_MUTED}; font-size: 9pt; font-weight: normal;"
            " border: none; background-color: transparent;"
        )
        path_label.setWordWrap(True)
        layout.addWidget(path_label)

        # Keyboard accessibility
        widget.setFocusPolicy(Qt.StrongFocus)
        widget.setCursor(Qt.PointingHandCursor)
        widget.setAccessibleName(f"Recent project: {project_path.stem}")

        # Connect click signal
        def on_project_open(path=project_path):
            logger.debug(
                "Quick action triggered",
                action="open_recent_project",
                page="HomePage",
                project_path=str(path),
            )
            self.recent_project_opened.emit(path)

        widget.clicked.connect(on_project_open)

        return widget

    def _create_system_status(self) -> QWidget:
        """Create system status section."""
        group = QGroupBox("System Status")

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

        layout = QVBoxLayout(group)
        layout.setContentsMargins(
            Spacing.LG, Spacing.XL + Spacing.SM, Spacing.LG, Spacing.LG
        )
        layout.setSpacing(Spacing.SM)

        resources = [
            ("Documentation", "https://rheojax.readthedocs.io"),
            ("Tutorials", "https://github.com/imewei/rheojax/tree/main/examples"),
            ("Report Issues", "https://github.com/imewei/rheojax/issues"),
        ]

        for title, url in resources:
            btn = QPushButton(title)
            btn.setProperty("variant", "secondary")
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(button_style("secondary", "md"))
            btn.clicked.connect(lambda checked, u=url, t=title: self._open_url(u, t))
            layout.addWidget(btn)

        layout.addStretch()

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

