"""Reusable empty state widget for placeholder content.

Styling is handled entirely via QSS property selectors so that colors
update automatically on live theme switches (light ↔ dark).  The base.qss
rules for ``QLabel[class="empty-message"]`` and ``QLabel[class="empty-desc"]``
provide the color/font-size definitions.
"""

from collections.abc import Callable

from rheojax.gui.compat import QLabel, QPushButton, Qt, QVBoxLayout, QWidget, Signal
from rheojax.gui.resources.styles.tokens import Spacing


class EmptyStateWidget(QWidget):
    """Centered empty state with optional action button.

    Replaces ad-hoc empty-state QLabels scattered across pages
    with a consistent, reusable component.

    Parameters
    ----------
    message : str
        Primary message (shown prominently).
    description : str, optional
        Secondary description text (shown smaller, muted).
    action_text : str, optional
        If provided, shows a clickable action button.
    parent : QWidget, optional
        Parent widget.
    """

    action_clicked = Signal()

    def __init__(
        self,
        message: str,
        description: str = "",
        action_text: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(Spacing.SM)
        layout.setContentsMargins(Spacing.XL, Spacing.XL, Spacing.XL, Spacing.XL)

        self._message_label = QLabel(message)
        self._message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._message_label.setWordWrap(True)
        self._message_label.setProperty("class", "empty-message")
        layout.addWidget(self._message_label)

        if description:
            self._desc_label = QLabel(description)
            self._desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._desc_label.setWordWrap(True)
            self._desc_label.setProperty("class", "empty-desc")
            layout.addWidget(self._desc_label)

        if action_text:
            self._action_btn = QPushButton(action_text)
            self._action_btn.setProperty("variant", "secondary")
            self._action_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._action_btn.clicked.connect(self.action_clicked)
            layout.addWidget(self._action_btn, alignment=Qt.AlignmentFlag.AlignCenter)

    def set_message(self, text: str) -> None:
        """Update the primary message text."""
        self._message_label.setText(text)

    def set_description(self, text: str) -> None:
        """Update the description text (if present)."""
        if hasattr(self, "_desc_label"):
            self._desc_label.setText(text)

    def set_action_button(self, label: str, callback: Callable[[], None]) -> None:
        """Set a primary call-to-action button on the empty state.

        If the button already exists (e.g. the instance was constructed with
        ``action_text``), its text and connection are updated in-place.
        Otherwise a new button is appended to the layout.

        Parameters
        ----------
        label : str
            Button label text.
        callback : Callable[[], None]
            Slot to invoke when the button is clicked.
        """
        if hasattr(self, "_action_btn"):
            self._action_btn.setText(label)
            # Disconnect the generic action_clicked relay and connect directly
            try:
                self._action_btn.clicked.disconnect()
            except RuntimeError:
                pass
            self._action_btn.clicked.connect(callback)
            self._action_btn.clicked.connect(self.action_clicked.emit)
        else:
            self._action_btn = QPushButton(label)
            self._action_btn.setProperty("variant", "secondary")
            self._action_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._action_btn.clicked.connect(callback)
            self._action_btn.clicked.connect(self.action_clicked.emit)
            self.layout().addWidget(
                self._action_btn, alignment=Qt.AlignmentFlag.AlignCenter
            )


__all__ = ["EmptyStateWidget"]
