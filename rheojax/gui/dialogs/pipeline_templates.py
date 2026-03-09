"""
Pipeline Templates Dialog
==========================

Dialog for selecting a pre-built pipeline template.  Displays a list of
available templates with descriptions and a live YAML preview.
"""

from __future__ import annotations

from rheojax.gui.compat import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QSplitter,
    Qt,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from rheojax.gui.resources.styles.tokens import Spacing
from rheojax.logging import get_logger

logger = get_logger(__name__)


class PipelineTemplateDialog(QDialog):
    """Dialog for selecting a pipeline template.

    Shows all registered templates in a list on the left and a read-only
    YAML preview on the right.  The caller receives the chosen template name
    (or ``None`` if cancelled) via :meth:`get_template`.

    Attributes
    ----------
    selected_template : str | None
        Name of the chosen template after the dialog is accepted,
        or ``None`` if the dialog was cancelled.

    Example
    -------
    >>> name = PipelineTemplateDialog.get_template()  # doctest: +SKIP
    >>> if name:  # doctest: +SKIP
    ...     yaml_str = get_template(name)  # doctest: +SKIP
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        logger.debug("Initializing", class_name=self.__class__.__name__)

        self.selected_template: str | None = None
        self._templates: list[dict[str, str]] = []

        self.setWindowTitle("Pipeline Templates")
        self.setMinimumSize(640, 480)

        self._setup_ui()
        self._load_templates()

        logger.debug("Initialization complete", class_name=self.__class__.__name__)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        """Build the dialog layout."""
        root = QVBoxLayout(self)
        root.setContentsMargins(Spacing.MD, Spacing.MD, Spacing.MD, Spacing.MD)
        root.setSpacing(Spacing.SM)

        # Instruction label
        instruction = QLabel(
            "Select a template to use as the starting point for your pipeline."
        )
        instruction.setStyleSheet("font-size: 10pt; color: gray;")
        instruction.setWordWrap(True)
        root.addWidget(instruction)

        # Splitter: template list (left) | YAML preview (right)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        # Left: template list
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(Spacing.XS)

        list_label = QLabel("Templates")
        list_label.setStyleSheet("font-size: 10pt; font-weight: bold;")
        left_layout.addWidget(list_label)

        self.template_list = QListWidget()
        self.template_list.setMinimumWidth(200)
        self.template_list.currentItemChanged.connect(self._on_selection_changed)
        left_layout.addWidget(self.template_list)

        splitter.addWidget(left_widget)

        # Right: YAML preview
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(Spacing.XS)

        preview_label = QLabel("Preview")
        preview_label.setStyleSheet("font-size: 10pt; font-weight: bold;")
        right_layout.addWidget(preview_label)

        self.preview_edit = QTextEdit()
        self.preview_edit.setReadOnly(True)
        self.preview_edit.setStyleSheet(
            "font-family: monospace; font-size: 10pt;"
        )
        self.preview_edit.setPlaceholderText("Select a template to preview its YAML...")
        right_layout.addWidget(self.preview_edit)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        root.addWidget(splitter, stretch=1)

        # Description row beneath the splitter
        self.description_label = QLabel("")
        self.description_label.setStyleSheet("font-size: 10pt; color: gray;")
        self.description_label.setWordWrap(True)
        root.addWidget(self.description_label)

        # Button box
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText(
            "Use Template"
        )
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self.button_box.accepted.connect(self._on_accept)
        self.button_box.rejected.connect(self._on_reject)
        root.addWidget(self.button_box)

    # ------------------------------------------------------------------
    # Template loading
    # ------------------------------------------------------------------

    def _load_templates(self) -> None:
        """Populate the list widget from the CLI template registry."""
        try:
            from rheojax.cli._templates import list_templates

            self._templates = list_templates()
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Could not load templates",
                class_name=self.__class__.__name__,
                error=str(exc),
            )
            self._templates = []

        self.template_list.clear()
        for tmpl in self._templates:
            item = QListWidgetItem(tmpl["name"])
            item.setToolTip(tmpl["description"])
            self.template_list.addItem(item)

        logger.debug(
            "Templates loaded",
            class_name=self.__class__.__name__,
            count=len(self._templates),
        )

        # Auto-select the first entry if available
        if self.template_list.count() > 0:
            self.template_list.setCurrentRow(0)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_selection_changed(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        """Update the YAML preview and description when selection changes."""
        if current is None:
            self.preview_edit.clear()
            self.description_label.setText("")
            self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(
                False
            )
            return

        name = current.text()
        description = ""
        for tmpl in self._templates:
            if tmpl["name"] == name:
                description = tmpl.get("description", "")
                break

        self.description_label.setText(description)

        try:
            from rheojax.cli._templates import get_template

            yaml_str = get_template(name)
            self.preview_edit.setPlainText(yaml_str)
        except KeyError:  # pragma: no cover
            self.preview_edit.setPlainText(f"# Template '{name}' not found.")

        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)

        logger.debug(
            "Template selected",
            dialog=self.__class__.__name__,
            template=name,
        )

    def _on_accept(self) -> None:
        """Store selected template name and close with Accepted result."""
        current = self.template_list.currentItem()
        if current is not None:
            self.selected_template = current.text()
        logger.debug(
            "Dialog closed",
            dialog=self.__class__.__name__,
            result="accepted",
            template=self.selected_template,
        )
        self.accept()

    def _on_reject(self) -> None:
        """Close with Rejected result; selected_template remains None."""
        logger.debug(
            "Dialog closed", dialog=self.__class__.__name__, result="rejected"
        )
        self.reject()

    # ------------------------------------------------------------------
    # Static convenience factory
    # ------------------------------------------------------------------

    @staticmethod
    def get_template(parent: QWidget | None = None) -> str | None:
        """Show the dialog and return the selected template name.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget for the dialog.

        Returns
        -------
        str | None
            The chosen template name (e.g. ``"basic"``), or ``None`` if
            the user cancelled.

        Example
        -------
        >>> name = PipelineTemplateDialog.get_template()  # doctest: +SKIP
        """
        dialog = PipelineTemplateDialog(parent)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.selected_template
        return None
