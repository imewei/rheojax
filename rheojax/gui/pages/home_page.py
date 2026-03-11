"""
Home Page
=========

Landing page with quick start actions, recent projects, and system status.
Redesigned with a compact hero, card-based quick-start grid, and a
two-column body layout for better information hierarchy.
"""

from pathlib import Path

from rheojax.gui.compat import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    Qt,
    QTimer,
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
    card_style,
    section_header_style,
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
        - Compact hero header with version + model count
        - Card-based quick start actions (Import, New, Open)
        - Recent projects list
        - System status (JAX device, float64, memory)
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

        # Compact hero header
        content_layout.addWidget(self._create_header())

        # Body: two-column layout
        body = QWidget()
        body.setStyleSheet(f"background-color: {themed('BG_CANVAS')};")
        body_layout = QHBoxLayout(body)
        body_layout.setSpacing(Spacing.XL)
        body_layout.setContentsMargins(
            Spacing.PAGE_MARGIN + 4,
            Spacing.XL,
            Spacing.PAGE_MARGIN + 4,
            Spacing.PAGE_MARGIN,
        )

        # Left column (primary): Quick Start + Recent Projects
        left_col = QVBoxLayout()
        left_col.setSpacing(Spacing.XL)
        left_col.addWidget(self._create_quick_start())
        left_col.addWidget(self._create_recent_projects())
        left_col.addStretch()

        # Right column (supporting): System Status + Resources
        right_col = QVBoxLayout()
        right_col.setSpacing(Spacing.XL)
        right_col.addWidget(self._create_system_status())
        right_col.addWidget(self._create_resources())
        right_col.addStretch()

        # 3:2 ratio — primary content gets more space
        body_layout.addLayout(left_col, stretch=3)
        body_layout.addLayout(right_col, stretch=2)

        content_layout.addWidget(body)

        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)

        # Defer model registry discovery to avoid blocking GUI startup
        QTimer.singleShot(0, self._populate_stat_pills)

    def _populate_stat_pills(self) -> None:
        """Populate header stat pills after event loop starts."""
        stats = self._get_inventory_stats()
        # Insert pills before the stretch item (last item in the layout)
        insert_pos = self._stat_pills_layout.count() - 1
        for label_text in stats:
            pill = QLabel(label_text)
            pill.setStyleSheet(self._stat_pill_style)
            self._stat_pills_layout.insertWidget(insert_pos, pill)
            insert_pos += 1

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    def _create_header(self) -> QWidget:
        """Create compact hero header with gradient background."""
        header = QWidget()
        header.setMinimumHeight(140)
        header.setMaximumHeight(160)
        header.setStyleSheet(f"""
            QWidget {{
                background-color: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 {themed('PRIMARY_PRESSED')},
                    stop:0.4 {themed('PRIMARY_PRESSED')},
                    stop:1 {themed('ACCENT_PRESSED')}
                );
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            }}
            QLabel {{ background-color: transparent; }}
        """)

        layout = QVBoxLayout(header)
        layout.setSpacing(Spacing.SM)
        layout.setContentsMargins(
            Spacing.PAGE_MARGIN + 16,
            Spacing.XL,
            Spacing.PAGE_MARGIN + 16,
            Spacing.LG,
        )

        # Top row: Title + Version badge + Stats
        top_row = QHBoxLayout()
        top_row.setSpacing(Spacing.MD)
        top_row.setContentsMargins(0, 0, 0, 0)

        # Title
        title = QLabel("RheoJAX")
        title.setStyleSheet(f"""
            color: {ColorPalette.TEXT_INVERSE};
            font-family: {Typography.FONT_FAMILY};
            font-size: {Typography.SIZE_HEADING + 4}pt;
            font-weight: {Typography.WEIGHT_BOLD};
            background-color: transparent;
        """)
        top_row.addWidget(title)

        # Version badge
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
                border-radius: {BorderRadius.SM}px;
                padding: {Spacing.XXS}px {Spacing.SM}px;
                font-size: {Typography.SIZE_SM}pt;
                font-weight: {Typography.WEIGHT_SEMIBOLD};
                font-family: {Typography.FONT_FAMILY_MONO};
            }}
        """)
        top_row.addWidget(version_badge)

        # Stat pills (model count + transforms) — populated asynchronously
        # to avoid blocking the GUI with model registry discovery.
        self._stat_pill_style = f"""
            QLabel {{
                color: rgba(255, 255, 255, 0.85);
                background-color: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: {BorderRadius.SM}px;
                padding: {Spacing.XXS}px {Spacing.SM}px;
                font-size: {Typography.SIZE_SM}pt;
                font-weight: {Typography.WEIGHT_MEDIUM};
                font-family: {Typography.FONT_FAMILY};
            }}
        """
        self._stat_pills_layout = top_row

        top_row.addStretch()
        layout.addLayout(top_row)

        # Subtitle
        subtitle = QLabel("JAX-Accelerated Rheological Analysis")
        subtitle.setStyleSheet(f"""
            color: rgba(255, 255, 255, 0.7);
            font-family: {Typography.FONT_FAMILY};
            font-size: {Typography.SIZE_LG}pt;
            font-weight: {Typography.WEIGHT_NORMAL};
            background-color: transparent;
        """)
        layout.addWidget(subtitle)

        layout.addStretch()

        return header

    @staticmethod
    def _get_inventory_stats() -> list[str]:
        """Return stat strings for the header pills."""
        try:
            import rheojax.models
            import rheojax.transforms

            rheojax.models._ensure_all_registered()
            rheojax.transforms._ensure_all_registered()

            from rheojax.core.registry import Registry

            registry = Registry.get_instance()
            inv = registry.inventory()
            n_models = len(inv.get("all_models", []))
            n_transforms = len(inv.get("all_transforms", []))
            pills = []
            if n_models:
                pills.append(f"{n_models} Models")
            if n_transforms:
                pills.append(f"{n_transforms} Transforms")
            return pills
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Quick Start
    # ------------------------------------------------------------------

    def _create_quick_start(self) -> QWidget:
        """Create quick start section with card grid."""
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(Spacing.MD)

        # Section label
        label = QLabel("Quick Start")
        label.setStyleSheet(section_header_style())
        outer.addWidget(label)

        # Card grid (3 columns)
        grid = QGridLayout()
        grid.setSpacing(Spacing.MD)
        grid.setContentsMargins(0, 0, 0, 0)

        cards = [
            {
                "title": "Import Data",
                "desc": "TRIOS, Anton Paar, CSV, Excel",
                "variant": "success",
                "slot": self._on_import_data_clicked,
            },
            {
                "title": "New Project",
                "desc": "Start a new analysis",
                "variant": "warning",
                "slot": self._on_new_project_clicked,
            },
            {
                "title": "Open Project",
                "desc": "Load existing project",
                "variant": "primary",
                "slot": self._on_open_project_clicked,
            },
        ]

        for col, card_info in enumerate(cards):
            card = self._create_action_card(
                card_info["title"],
                card_info["desc"],
                card_info["variant"],
            )
            card.clicked.connect(card_info["slot"])
            grid.addWidget(card, 0, col)

        outer.addLayout(grid)
        return container

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

    def _create_action_card(
        self, title: str, description: str, variant: str
    ) -> ClickableWidget:
        """Create a quick-start action card with title, description, and color accent."""
        card = ClickableWidget()
        card.setFocusPolicy(Qt.StrongFocus)
        card.setCursor(Qt.PointingHandCursor)
        card.setAccessibleName(f"Quick start: {title}")
        card.setMinimumHeight(96)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Accent color bar at the top of the card
        accent_colors = {
            "primary": themed("PRIMARY"),
            "success": themed("SUCCESS"),
            "warning": themed("WARNING"),
            "error": themed("ERROR"),
            "accent": themed("ACCENT"),
        }
        accent = accent_colors.get(variant, themed("PRIMARY"))

        card.setStyleSheet(f"""
            QWidget {{
                background-color: {themed('BG_ELEVATED')};
                border: 1px solid {themed('BORDER_SUBTLE')};
                border-top: 3px solid {accent};
                border-radius: {BorderRadius.LG}px;
            }}
            QWidget:hover {{
                background-color: {themed('BG_HOVER')};
                border-color: {themed('BORDER_DEFAULT')};
                border-top: 3px solid {accent};
            }}
            QLabel {{ background-color: transparent; border: none; }}
        """)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(Spacing.LG, Spacing.LG, Spacing.LG, Spacing.LG)
        layout.setSpacing(Spacing.XS)

        title_label = QLabel(title)
        title_label.setStyleSheet(f"""
            font-family: {Typography.FONT_FAMILY};
            font-size: {Typography.SIZE_MD}pt;
            font-weight: {Typography.WEIGHT_SEMIBOLD};
            color: {themed('TEXT_PRIMARY')};
        """)
        layout.addWidget(title_label)

        desc_label = QLabel(description)
        desc_label.setStyleSheet(f"""
            font-family: {Typography.FONT_FAMILY};
            font-size: {Typography.SIZE_SM}pt;
            font-weight: {Typography.WEIGHT_NORMAL};
            color: {themed('TEXT_MUTED')};
        """)
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        layout.addStretch()

        return card

    # ------------------------------------------------------------------
    # Recent Projects
    # ------------------------------------------------------------------

    def _create_recent_projects(self) -> QWidget:
        """Create recent projects section."""
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(Spacing.SM)

        label = QLabel("Recent Projects")
        label.setStyleSheet(section_header_style())
        outer.addWidget(label)

        # Get recent projects from state
        recent_projects = self._store.get_state().recent_projects

        if not recent_projects:
            empty = QLabel("No recent projects. Import data or open a project to get started.")
            empty.setAlignment(Qt.AlignCenter)
            empty.setStyleSheet(f"""
                color: {themed('TEXT_MUTED')};
                font-size: {Typography.SIZE_MD_SM}pt;
                padding: {Spacing.XL}px {Spacing.LG}px;
                background-color: {themed('BG_SURFACE')};
                border: 1px dashed {themed('BORDER_SUBTLE')};
                border-radius: {BorderRadius.LG}px;
            """)
            empty.setWordWrap(True)
            outer.addWidget(empty)
        else:
            for project_path in recent_projects[:5]:
                outer.addWidget(self._create_recent_project_item(project_path))

        return container

    def _create_recent_project_item(self, project_path: Path) -> QWidget:
        """Create a recent project item."""
        widget = ClickableWidget()
        widget.setStyleSheet(f"""
            QWidget {{
                background-color: {themed('BG_ELEVATED')};
                border: 1px solid {themed('BORDER_SUBTLE')};
                border-radius: {BorderRadius.LG}px;
            }}
            QWidget:hover {{
                background-color: {themed('BG_HOVER')};
                border-color: {themed('BORDER_DEFAULT')};
            }}
            QLabel {{ background-color: transparent; border: none; }}
        """)

        layout = QHBoxLayout(widget)
        layout.setContentsMargins(Spacing.MD, Spacing.SM, Spacing.MD, Spacing.SM)
        layout.setSpacing(Spacing.SM)

        # Left: name + path stacked
        text_col = QVBoxLayout()
        text_col.setSpacing(Spacing.XXS)

        name = QLabel(project_path.stem)
        name.setStyleSheet(f"""
            font-weight: {Typography.WEIGHT_SEMIBOLD};
            font-size: {Typography.SIZE_MD_SM}pt;
            color: {themed('TEXT_PRIMARY')};
        """)
        text_col.addWidget(name)

        path_label = QLabel(str(project_path.parent))
        path_label.setStyleSheet(f"""
            color: {themed('TEXT_MUTED')};
            font-size: {Typography.SIZE_XS}pt;
            font-weight: {Typography.WEIGHT_NORMAL};
        """)
        path_label.setWordWrap(True)
        text_col.addWidget(path_label)

        layout.addLayout(text_col, stretch=1)

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

    # ------------------------------------------------------------------
    # System Status
    # ------------------------------------------------------------------

    def _create_system_status(self) -> QWidget:
        """Create system status section."""
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(Spacing.SM)

        label = QLabel("System Status")
        label.setStyleSheet(section_header_style())
        outer.addWidget(label)

        # Card wrapper for the JAX status widget
        card = QWidget()
        card.setStyleSheet(f"""
            QWidget {{
                {card_style()}
            }}
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(
            Spacing.LG, Spacing.LG, Spacing.LG, Spacing.LG
        )

        self._jax_status = JAXStatusWidget()
        card_layout.addWidget(self._jax_status)

        outer.addWidget(card)

        # Initialize JAX status
        self._update_jax_status()

        return container

    def _update_jax_status(self) -> None:
        """Update JAX status display."""
        try:
            from rheojax.gui.utils.jax_utils import get_jax_info

            jax_info = get_jax_info()

            devices = jax_info.get("devices", ["cpu"])
            self._jax_status.update_device_list(devices)

            current_device = jax_info.get("default_device", "cpu")
            self._jax_status.set_current_device(current_device)

            memory_used = jax_info.get("memory_used_mb", 0)
            memory_total = jax_info.get("memory_total_mb", 0)
            self._jax_status.update_memory(memory_used, memory_total)

            float64_enabled = jax_info.get("float64_enabled", False)
            self._jax_status.set_float64_enabled(float64_enabled)

            jit_cache_count = jax_info.get("jit_cache_count", 0)
            self._jax_status.update_jit_cache(jit_cache_count)

        except Exception:
            logger.error("Failed to update JAX status", exc_info=True, page="HomePage")
            self._jax_status.update_device_list(["cpu"])
            self._jax_status.set_current_device("cpu")
            self._jax_status.set_float64_enabled(False)

    # ------------------------------------------------------------------
    # Resources
    # ------------------------------------------------------------------

    def _create_resources(self) -> QWidget:
        """Create resources section."""
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(Spacing.SM)

        label = QLabel("Resources")
        label.setStyleSheet(section_header_style())
        outer.addWidget(label)

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
            outer.addWidget(btn)

        return container

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
