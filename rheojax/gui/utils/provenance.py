"""
Provenance Tracking
==================

Track analysis provenance and metadata.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from rheojax.logging import get_logger

logger = get_logger(__name__)


class Provenance:
    """Provenance tracking for reproducibility.

    Features:
        - Operation history
        - Parameter tracking
        - Timestamp logging
        - Dependency versions

    Example
    -------
    >>> prov = Provenance()  # doctest: +SKIP
    >>> prov.record('fit', model='maxwell', dataset='ds1')  # doctest: +SKIP
    >>> history = prov.get_history()  # doctest: +SKIP
    """

    def __init__(self) -> None:
        """Initialize provenance tracker."""
        logger.debug("Initializing Provenance tracker")
        self._history: list[dict[str, Any]] = []
        self._created_at = datetime.now().isoformat()
        logger.debug("Provenance tracker initialized", created_at=self._created_at)

    def record(
        self,
        operation: str,
        **metadata: Any,
    ) -> str:
        """Record operation with metadata.

        Parameters
        ----------
        operation : str
            Operation name
        **metadata
            Operation metadata

        Returns
        -------
        str
            Record ID
        """
        logger.debug(
            "Recording operation",
            operation=operation,
            metadata_keys=list(metadata.keys()),
        )
        record_id = str(uuid.uuid4())[:8]
        record = {
            "id": record_id,
            "operation": operation,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata,
        }
        self._history.append(record)
        logger.debug(
            "Operation recorded",
            record_id=record_id,
            operation=operation,
            history_length=len(self._history),
        )
        return record_id

    def get_history(self) -> list[dict[str, Any]]:
        """Get operation history.

        Returns
        -------
        list[dict]
            Chronological operation records
        """
        logger.debug("Getting operation history", history_length=len(self._history))
        return list(self._history)

    def export(self, file_path: str) -> None:
        """Export provenance to file.

        Parameters
        ----------
        file_path : str
            Output file path
        """
        logger.debug("Entering export", file_path=file_path)
        try:
            path = Path(file_path)
            export_data = {
                "created_at": self._created_at,
                "exported_at": datetime.now().isoformat(),
                "history": self._history,
            }
            with path.open("w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, default=str)
            logger.debug(
                "Provenance exported successfully",
                file_path=file_path,
                record_count=len(self._history),
            )
        except Exception as e:
            logger.error(
                "Failed to export provenance",
                file_path=file_path,
                exc_info=True,
            )
            raise

    def clear(self) -> None:
        """Clear operation history."""
        logger.debug("Clearing operation history", previous_length=len(self._history))
        self._history.clear()
        logger.debug("Operation history cleared")

    def __len__(self) -> int:
        """Return number of recorded operations."""
        return len(self._history)
