"""JSON envelope dataclass for stdin/stdout piping between CLI commands.

RheoJAX CLI commands can be composed in shell pipelines::

    rheojax fit data.csv --model maxwell --json | rheojax bayesian --json

Each command reads an optional upstream ``Envelope`` from stdin and writes
its own ``Envelope`` to stdout.  This module defines the envelope schema
and helpers for encoding/decoding.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any

from rheojax.io.json_encoder import NumpyJSONEncoder
from rheojax.logging import get_logger

logger = get_logger(__name__)

# Maximum bytes to read from stdin to prevent OOM from malicious input
MAX_STDIN_BYTES = 256 * 1024 * 1024  # 256 MB


def _get_version() -> str:
    """Return the installed rheojax version string."""
    import rheojax

    return rheojax.__version__


@dataclass
class Envelope:
    """Structured JSON envelope exchanged between piped CLI commands.

    Attributes:
        rheojax_version: Version of the rheojax package that produced the envelope.
        envelope_type: Category of payload — ``"data"``, ``"fit_result"``,
            or ``"bayesian_result"``.
        data: Raw data payload (used when ``envelope_type == "data"``).
        fit_result: Fit result payload (used when ``envelope_type == "fit_result"``).
        bayesian_result: Bayesian result payload.
        metadata: Arbitrary key/value annotations (test_mode, model name, etc.).
    """

    rheojax_version: str
    envelope_type: str
    data: dict[str, Any] | None = None
    fit_result: dict[str, Any] | None = None
    bayesian_result: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        """Serialise envelope to a JSON string.

        Returns:
            Compact JSON string (no unnecessary whitespace).

        Example:
            >>> env = Envelope("0.6.0", "data", data={"x": [1, 2]})
            >>> json_str = env.to_json()
        """
        payload: dict[str, Any] = {
            "rheojax_version": self.rheojax_version,
            "envelope_type": self.envelope_type,
            "metadata": self.metadata,
        }
        if self.data is not None:
            payload["data"] = self.data
        if self.fit_result is not None:
            payload["fit_result"] = self.fit_result
        if self.bayesian_result is not None:
            payload["bayesian_result"] = self.bayesian_result

        return json.dumps(payload, cls=NumpyJSONEncoder, separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> Envelope:
        """Deserialise an ``Envelope`` from a JSON string.

        Args:
            s: JSON string produced by :meth:`to_json`.

        Returns:
            Reconstructed :class:`Envelope`.

        Raises:
            ValueError: If the JSON is missing required fields.

        Example:
            >>> env = Envelope.from_json(json_str)
        """
        try:
            raw: dict[str, Any] = json.loads(s)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid envelope JSON: {exc}") from exc

        if "rheojax_version" not in raw or "envelope_type" not in raw:
            raise ValueError(
                "Envelope JSON must contain 'rheojax_version' and 'envelope_type' keys."
            )

        return cls(
            rheojax_version=raw["rheojax_version"],
            envelope_type=raw["envelope_type"],
            data=raw.get("data"),
            fit_result=raw.get("fit_result"),
            bayesian_result=raw.get("bayesian_result"),
            metadata=raw.get("metadata", {}),
        )

    # ------------------------------------------------------------------
    # stdin / stdout helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_stdin(cls) -> Envelope | None:
        """Read an envelope from stdin if stdin is not a terminal.

        Returns:
            Parsed :class:`Envelope`, or ``None`` if stdin is a tty or empty.

        Example:
            >>> upstream = Envelope.from_stdin()
        """
        if sys.stdin.isatty():
            logger.debug("stdin is a tty — no upstream envelope")
            return None

        raw = sys.stdin.read(MAX_STDIN_BYTES).strip()
        if not raw:
            logger.debug("stdin was empty — no upstream envelope")
            return None

        try:
            envelope = cls.from_json(raw)
            logger.debug(
                "Read upstream envelope from stdin",
                envelope_type=envelope.envelope_type,
                version=envelope.rheojax_version,
            )
            return envelope
        except ValueError as exc:
            print(f"Warning: Invalid pipe input — {exc}", file=sys.stderr)
            logger.warning("Failed to parse stdin envelope", error=str(exc))
            return None

    def write_stdout(self) -> None:
        """Write the envelope as JSON to stdout.

        The newline ensures downstream processes can detect message
        boundaries when reading line-by-line.

        Example:
            >>> env.write_stdout()
        """
        sys.stdout.write(self.to_json())
        sys.stdout.write("\n")
        sys.stdout.flush()
        logger.debug("Wrote envelope to stdout", envelope_type=self.envelope_type)


# ------------------------------------------------------------------
# Factory helpers
# ------------------------------------------------------------------


def create_data_envelope(
    x: Any,
    y: Any,
    metadata: dict[str, Any] | None = None,
) -> Envelope:
    """Create a ``"data"`` envelope from x/y arrays.

    Args:
        x: Independent variable array (will be JSON-serialised via
           :class:`~rheojax.io.json_encoder.NumpyJSONEncoder`).
        y: Dependent variable array.
        metadata: Optional annotations (e.g. ``{"test_mode": "relaxation"}``).

    Returns:
        Configured :class:`Envelope` with ``envelope_type="data"``.

    Example:
        >>> import numpy as np
        >>> env = create_data_envelope(np.linspace(0, 1, 10), np.ones(10))
    """

    def _coerce(arr: Any) -> list:
        if hasattr(arr, "tolist"):
            return arr.tolist()
        if hasattr(arr, "__iter__"):
            return list(arr)
        return arr

    return Envelope(
        rheojax_version=_get_version(),
        envelope_type="data",
        data={"x": _coerce(x), "y": _coerce(y)},
        metadata=metadata or {},
    )


def create_fit_envelope(
    model: Any,
    params: dict[str, Any],
    test_mode: str,
    metadata: dict[str, Any] | None = None,
) -> Envelope:
    """Create a ``"fit_result"`` envelope from a fitted model.

    Args:
        model: Fitted rheojax model instance (used to extract the model name).
        params: Mapping of parameter names to fitted values.
        test_mode: Protocol identifier (e.g. ``"relaxation"``, ``"oscillation"``).
        metadata: Optional extra annotations.

    Returns:
        Configured :class:`Envelope` with ``envelope_type="fit_result"``.

    Example:
        >>> env = create_fit_envelope(model, {"G_e": 1000.0, "tau": 0.1}, "relaxation")
    """
    model_name: str = getattr(model, "name", type(model).__name__)

    fit_result: dict[str, Any] = {
        "model": model_name,
        "test_mode": test_mode,
        "parameters": {k: float(v) if hasattr(v, "item") else v for k, v in params.items()},
    }

    return Envelope(
        rheojax_version=_get_version(),
        envelope_type="fit_result",
        fit_result=fit_result,
        metadata=metadata or {},
    )
