"""
Workspace Container
===================

QStackedWidget that maps pipeline step types to existing page classes.

Swaps workspace panels based on the selected pipeline step type, defaulting
to the HomePage when no step is selected.
"""

from rheojax.gui.compat import QStackedWidget, QWidget, Signal
from rheojax.logging import get_logger

logger = get_logger(__name__)


class WorkspaceContainer(QStackedWidget):
    """Swaps workspace panels based on the selected pipeline step type.

    Pages are lazily imported and instantiated on first use to keep startup
    time low. The default page (HomePage) is shown when no step is selected.

    Step Type Mapping
    -----------------
    None          -> HomePage
    "load"        -> DataPage
    "transform"   -> TransformPage
    "fit"         -> FitPage
    "bayesian"    -> BayesianPage
    "export"      -> ExportPage

    Signals
    -------
    page_changed : Signal(str)
        Emitted after ``show_step()`` switches to a new page, carrying the
        resolved step-type key (e.g. ``"fit"``) or ``"home"`` for the
        HomePage.  The main window can log page transitions without coupling
        to page internals.

    Example
    -------
    >>> container = WorkspaceContainer()  # doctest: +SKIP
    >>> container.show_step("fit")  # doctest: +SKIP
    >>> page = container.get_current_page()  # doctest: +SKIP
    """

    page_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the workspace container.

        Instantiates all page classes and registers them in the stack.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        logger.debug("Initializing", class_name=self.__class__.__name__)

        # Lazy import page classes to avoid heavy upfront imports
        from rheojax.gui.pages import (
            BayesianPage,
            DataPage,
            DiagnosticsPage,
            ExportPage,
            FitPage,
            HomePage,
            TransformPage,
        )

        # Instantiate pages — each takes parent=self
        self._home_page = HomePage(parent=self)
        self._data_page = DataPage(parent=self)
        self._transform_page = TransformPage(parent=self)
        self._fit_page = FitPage(parent=self)
        self._bayesian_page = BayesianPage(parent=self)
        self._diagnostics_page = DiagnosticsPage(parent=self)
        self._export_page = ExportPage(parent=self)

        # Add to stack and record indices
        self._home_index = self.addWidget(self._home_page)
        self._page_index: dict[str, int] = {
            "load": self.addWidget(self._data_page),
            "transform": self.addWidget(self._transform_page),
            "fit": self.addWidget(self._fit_page),
            "bayesian": self.addWidget(self._bayesian_page),
            "diagnostics": self.addWidget(self._diagnostics_page),
            "export": self.addWidget(self._export_page),
        }

        # Start on HomePage
        self.setCurrentIndex(self._home_index)
        logger.debug("Initialization complete", class_name=self.__class__.__name__)

    def show_step(self, step_type: str | None) -> None:
        """Switch to the page for the given step type.

        Parameters
        ----------
        step_type : str | None
            Pipeline step type key, or None to show the home page.
            Unknown step types fall back to the home page.
        """
        if step_type is None or step_type not in self._page_index:
            logger.debug(
                "State updated",
                widget=self.__class__.__name__,
                action="show_step",
                step_type=step_type,
                resolved="home",
            )
            self.setCurrentIndex(self._home_index)
            self.page_changed.emit("home")
            return

        index = self._page_index[step_type]
        logger.debug(
            "State updated",
            widget=self.__class__.__name__,
            action="show_step",
            step_type=step_type,
            index=index,
        )
        self.setCurrentIndex(index)
        self.page_changed.emit(step_type)

    def get_current_page(self) -> QWidget:
        """Get the currently visible page widget.

        Returns
        -------
        QWidget
            The active page widget
        """
        return self.currentWidget()
