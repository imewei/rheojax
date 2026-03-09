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
from rheojax.gui.state.store import StepStatus
from rheojax.logging import get_logger

logger = get_logger(__name__)

QStyledItemDelegate = QtWidgets.QStyledItemDelegate
QPainter = QtGui.QPainter
# QRect lives in QtCore (not QtGui) in Qt5 and Qt6
QRect = QtCore.QRect

# Data roles for QListWidgetItem
ROLE_STEP_ID = Qt.UserRole
ROLE_STEP_TYPE = Qt.UserRole + 1
ROLE_STATUS = Qt.UserRole + 2

# Status color map
_STATUS_COLORS: dict[StepStatus, str] = {
    StepStatus.PENDING: "#9CA3AF",  # gray-400
    StepStatus.ACTIVE: "#3B82F6",  # blue-500
    StepStatus.COMPLETE: "#22C55E",  # green-500
    StepStatus.WARNING: "#F97316",  # orange-500
    StepStatus.ERROR: "#EF4444",  # red-500
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
        is_selected = bool(option.state & 0x0002)  # QStyle.State_Selected
        if is_selected:
            painter.fillRect(rect, QColor("#EEF2FF"))
        else:
            painter.fillRect(rect, QColor("transparent"))

        # Retrieve data
        step_type = index.data(ROLE_STEP_TYPE) or ""
        raw_status = index.data(ROLE_STATUS)
        step_name = index.data(Qt.DisplayRole) or ""

        # Resolve status — stored as StepStatus enum or its integer value
        status: StepStatus = StepStatus.PENDING
        if isinstance(raw_status, StepStatus):
            status = raw_status
        elif raw_status is not None:
            try:
                status = StepStatus(raw_status)
            except (ValueError, KeyError):
                pass

        color_hex = _STATUS_COLORS.get(status, "#9CA3AF")
        status_color = QColor(color_hex)

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
        pos_font.setPointSize(9)
        painter.setFont(pos_font)
        painter.setPen(QColor("#374151"))  # gray-700

        pos_rect = QRect(x, rect.top() + _PADDING_V, 20, rect.height() - 2 * _PADDING_V)
        painter.drawText(
            pos_rect,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            position_text,
        )
        x += 22

        # Draw step name
        name_font = QFont()
        name_font.setPointSize(9)
        painter.setFont(name_font)
        painter.setPen(QColor("#111827") if not is_selected else QColor("#1E1B4B"))

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
            type_font.setPointSize(8)
            painter.setFont(type_font)
            painter.setPen(QColor("#9CA3AF"))  # gray-400

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
