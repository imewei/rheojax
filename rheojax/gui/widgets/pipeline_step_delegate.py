"""
Pipeline Step Delegate
======================

Custom QStyledItemDelegate for rendering pipeline step items in a QListWidget.

Each item renders a status circle, position number, step name, and step type label.
"""

from rheojax.gui.compat import (
    QColor,
    QFont,
    QSize,
    Qt,
    QtCore,
    QtGui,
    QtWidgets,
    QWidget,
)
from rheojax.gui.resources.styles.tokens import Typography, themed
from rheojax.gui.state.store import StepStatus
from rheojax.logging import get_logger

logger = get_logger(__name__)

QStyledItemDelegate = QtWidgets.QStyledItemDelegate
QPainter = QtGui.QPainter
# QRect lives in QtCore (not QtGui) in Qt5 and Qt6
QRect = QtCore.QRect

# Data roles for QListWidgetItem (scoped enum for PySide6/Qt6 compatibility)
ROLE_STEP_ID = Qt.ItemDataRole.UserRole
ROLE_STEP_TYPE = Qt.ItemDataRole.UserRole + 1
ROLE_STATUS = Qt.ItemDataRole.UserRole + 2

# Shared pipeline-step status -> semantic color token mapping.
# Values are ColorPalette / DarkColorPalette attribute names, resolved via
# themed() at paint/style time so they track the active (light/dark) theme.
# Used by both PipelineStepDelegate and PipelineChips for a consistent palette.
STATUS_TOKENS: dict[StepStatus, str] = {
    StepStatus.PENDING: "TEXT_MUTED",
    StepStatus.ACTIVE: "PRIMARY",
    StepStatus.COMPLETE: "SUCCESS",
    StepStatus.WARNING: "WARNING",
    StepStatus.ERROR: "ERROR",
}

_CIRCLE_SIZE = 10
_ITEM_HEIGHT = 44
_PADDING_H = 10
_PADDING_V = 4


class PipelineStepDelegate(QStyledItemDelegate):
    """QStyledItemDelegate that renders pipeline step list items.

    Each row shows:
    - A 10px filled circle colored by StepStatus
    - A bold position number
    - The step name
    - A muted step_type label on the right

    Data Roles
    ----------
    Qt.UserRole     : step_id (str)
    Qt.UserRole + 1 : step_type (str)
    Qt.UserRole + 2 : status (StepStatus)

    Example
    -------
    >>> delegate = PipelineStepDelegate()  # doctest: +SKIP
    >>> list_widget.setItemDelegate(delegate)  # doctest: +SKIP
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the delegate.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)

    def sizeHint(self, option, index) -> QSize:  # noqa: ARG002
        """Return item size.

        Parameters
        ----------
        option : QStyleOptionViewItem
            Style options (unused)
        index : QModelIndex
            Item model index (unused)

        Returns
        -------
        QSize
            Preferred item size (200 x 44 px)
        """
        return QSize(200, _ITEM_HEIGHT)

    def paint(self, painter: QPainter, option, index) -> None:
        """Paint the item.

        Parameters
        ----------
        painter : QPainter
            Active painter
        option : QStyleOptionViewItem
            Style options
        index : QModelIndex
            Item model index
        """
        painter.save()

        rect = option.rect

        # Highlight selected items
        is_selected = bool(option.state & QtWidgets.QStyle.StateFlag.State_Selected)
        if is_selected:
            painter.fillRect(rect, QColor(themed("PRIMARY_SUBTLE")))
        else:
            painter.fillRect(rect, QColor("transparent"))

        # Retrieve data
        step_type = index.data(ROLE_STEP_TYPE) or ""
        raw_status = index.data(ROLE_STATUS)
        step_name = index.data(Qt.ItemDataRole.DisplayRole) or ""

        # Resolve status — stored as StepStatus enum or its integer value
        status: StepStatus = StepStatus.PENDING
        if isinstance(raw_status, StepStatus):
            status = raw_status
        elif raw_status is not None:
            try:
                status = StepStatus(raw_status)
            except (ValueError, KeyError):
                pass

        status_color = QColor(themed(STATUS_TOKENS.get(status, "TEXT_MUTED")))

        x = rect.left() + _PADDING_H
        cy = rect.top() + rect.height() // 2

        # Draw status circle
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(status_color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(x, cy - _CIRCLE_SIZE // 2, _CIRCLE_SIZE, _CIRCLE_SIZE)

        x += _CIRCLE_SIZE + 8

        # Draw position number (bold)
        position_text = str(index.row() + 1)
        pos_font = QFont()
        pos_font.setBold(True)
        pos_font.setPointSize(Typography.SIZE_XS)
        painter.setFont(pos_font)
        painter.setPen(QColor(themed("TEXT_SECONDARY")))

        pos_rect = QRect(x, rect.top() + _PADDING_V, 20, rect.height() - 2 * _PADDING_V)
        painter.drawText(
            pos_rect,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            position_text,
        )
        x += 22

        # Draw step name
        name_font = QFont()
        name_font.setPointSize(Typography.SIZE_XS)
        painter.setFont(name_font)
        painter.setPen(
            QColor(themed("PRIMARY")) if is_selected else QColor(themed("TEXT_PRIMARY"))
        )

        # Reserve right side for step_type label
        type_label_width = 58
        name_width = rect.right() - x - type_label_width - _PADDING_H
        name_rect = QRect(
            x,
            rect.top() + _PADDING_V,
            max(name_width, 0),
            rect.height() - 2 * _PADDING_V,
        )
        painter.drawText(
            name_rect,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            step_name,
        )

        # Draw step_type label (muted, right-aligned)
        if step_type:
            type_font = QFont()
            type_font.setPointSize(Typography.SIZE_XS)
            painter.setFont(type_font)
            painter.setPen(QColor(themed("TEXT_MUTED")))

            type_rect = QRect(
                rect.right() - type_label_width - _PADDING_H,
                rect.top() + _PADDING_V,
                type_label_width,
                rect.height() - 2 * _PADDING_V,
            )
            painter.drawText(
                type_rect,
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                step_type,
            )

        painter.restore()
