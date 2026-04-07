"""Multi-file loaders for TTS, SRFS, and generic series workflows."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rheojax.io.readers._utils import normalize_temperature
from rheojax.io.readers.auto import auto_load
from rheojax.logging import get_logger

if TYPE_CHECKING:
    from rheojax.core.data import RheoData

logger = get_logger(__name__)

__all__ = ["load_tts", "load_srfs", "load_series"]


def _validate_path_no_traversal(path: Path, *, label: str = "path") -> Path:
    """Resolve a path and reject it if it attempts directory traversal.

    Checks for ``..`` components in both the raw string and the resolved path
    to guard against encoded or Unicode-based traversal tricks.

    Returns the resolved (absolute) path.
    """
    raw = str(path)
    # Check raw string for any form of ".." (catches ....// and similar tricks)
    parts = Path(raw).parts
    if any(part == ".." for part in parts):
        logger.warning(
            "Path rejected: '..' traversal component detected",
            **{label: raw},
        )
        raise ValueError(
            f"{label.capitalize()} '{raw}' rejected: '..' path traversal "
            f"is not allowed."
        )
    return path.resolve()


def _expand_glob(files: list[str | Path] | str) -> list[Path]:
    """Expand a glob pattern or normalise a list of paths to sorted Path objects.

    All paths are resolved and checked for directory traversal.
    """
    if isinstance(files, str) and ("*" in files or "?" in files):
        _validate_path_no_traversal(Path(files), label="glob pattern")
        p = Path(files)
        expanded = sorted(p.parent.glob(p.name))
        if not expanded:
            raise FileNotFoundError(f"No files matched glob pattern: '{files}'")
        return [_validate_path_no_traversal(ep, label="expanded path") for ep in expanded]
    if isinstance(files, (str, Path)):
        return [_validate_path_no_traversal(Path(files), label="file path")]
    return [_validate_path_no_traversal(Path(f), label="file path") for f in files]


def _flatten_result(result: RheoData | list[RheoData]) -> RheoData:
    """Return a single RheoData from an auto_load result (take first if list)."""
    if isinstance(result, list):
        if len(result) > 1:
            warnings.warn(
                f"auto_load returned {len(result)} segments; using the first one. "
                "Pass return_all_segments=False or handle multi-segment files explicitly.",
                UserWarning,
                stacklevel=3,
            )
        return result[0]
    return result


def load_tts(
    files: list[str | Path] | str,
    T_ref: float,
    *,
    temperatures: list[float] | None = None,
    temperature_unit: str = "K",
    format: str | None = None,
    **kwargs: Any,
) -> list[RheoData]:
    """Load multiple files for a Time-Temperature Superposition (TTS) workflow.

    Each file corresponds to a single temperature. Files are loaded with
    :func:`auto_load` and tagged with temperature metadata, then sorted by
    temperature (ascending).

    Args:
        files: List of file paths **or** a glob pattern string (e.g.
            ``"data/T*.csv"``).
        T_ref: Reference temperature in Kelvin stored in metadata of every
            returned :class:`~rheojax.core.data.RheoData`.
        temperatures: Explicit temperature values (same length as *files*).
            Converted to Kelvin using *temperature_unit*. If ``None``, the
            function tries to read ``metadata["temperature"]`` from each loaded
            file.
        temperature_unit: Unit of *temperatures* — ``"K"`` (default), ``"C"``,
            or ``"F"``. Ignored when *temperatures* is ``None``.
        format: Optional format hint forwarded to :func:`auto_load`
            (``'trios'``, ``'anton_paar'``, ``'csv'``, ``'excel'``).
        **kwargs: Additional keyword arguments forwarded to :func:`auto_load`.

    Returns:
        List of :class:`~rheojax.core.data.RheoData` objects sorted by
        temperature (ascending).

    Raises:
        FileNotFoundError: If a glob pattern matches no files.
        ValueError: If *temperatures* length does not match the number of files,
            or if temperatures cannot be extracted from metadata.
    """
    paths = _expand_glob(files)

    if temperatures is not None and len(temperatures) != len(paths):
        raise ValueError(
            f"Length of 'temperatures' ({len(temperatures)}) does not match "
            f"the number of files ({len(paths)})."
        )

    # Convert provided temperatures to Kelvin up-front
    temps_K: list[float | None]
    if temperatures is not None:
        temps_K = [normalize_temperature(t, temperature_unit) for t in temperatures]
    else:
        temps_K = [None] * len(paths)

    results: list[RheoData] = []
    for i, path in enumerate(paths):
        logger.debug("load_tts: loading file", filepath=str(path), index=i)
        raw = auto_load(path, format=format, **kwargs)
        rd = _flatten_result(raw)

        # Assign or extract temperature
        if temps_K[i] is not None:
            rd.metadata["temperature"] = temps_K[i]
        else:
            # Try to read from existing metadata
            existing = rd.metadata.get("temperature") if rd.metadata else None
            if existing is None:
                raise ValueError(
                    f"No temperature found for file '{path}'. Either provide "
                    f"the 'temperatures' argument or ensure the file metadata "
                    f"contains a 'temperature' key."
                )
            # Normalise existing value — assume it is already in Kelvin unless
            # the metadata also carries a 'temperature_unit' hint.
            # All readers store temperature in Kelvin after conversion,
            # so default assumption is "K" if no unit hint is present.
            meta_unit = rd.metadata.get("temperature_unit", "K")
            rd.metadata["temperature"] = normalize_temperature(
                float(existing), meta_unit
            )

        rd.metadata["T_ref"] = T_ref
        results.append(rd)

    # Sort by temperature ascending
    results.sort(key=lambda r: r.metadata["temperature"])
    logger.debug("load_tts: loaded %d files, T_ref=%g K", len(results), T_ref)
    return results


def load_srfs(
    files: list[str | Path] | str,
    reference_gamma_dots: list[float],
    *,
    format: str | None = None,
    **kwargs: Any,
) -> list[RheoData]:
    """Load multiple files for a Superposition of Rate-Frequency Sweeps (SRFS) workflow.

    Each file corresponds to a different reference shear rate. Files are loaded
    with :func:`auto_load`, tagged with ``metadata["reference_gamma_dot"]``, and
    sorted by reference shear rate (ascending).

    Args:
        files: List of file paths or a glob pattern string.
        reference_gamma_dots: Reference shear rates (1/s) — one per file.
        format: Optional format hint forwarded to :func:`auto_load`.
        **kwargs: Additional keyword arguments forwarded to :func:`auto_load`.

    Returns:
        List of :class:`~rheojax.core.data.RheoData` objects sorted by
        ``reference_gamma_dot`` (ascending).

    Raises:
        FileNotFoundError: If a glob pattern matches no files.
        ValueError: If *reference_gamma_dots* length does not match the number
            of files.
    """
    paths = _expand_glob(files)

    if len(reference_gamma_dots) != len(paths):
        raise ValueError(
            f"Length of 'reference_gamma_dots' ({len(reference_gamma_dots)}) "
            f"does not match the number of files ({len(paths)})."
        )

    results: list[RheoData] = []
    for i, path in enumerate(paths):
        logger.debug("load_srfs: loading file", filepath=str(path), index=i)
        raw = auto_load(path, format=format, **kwargs)
        rd = _flatten_result(raw)
        rd.metadata["reference_gamma_dot"] = reference_gamma_dots[i]
        results.append(rd)

    # Sort by reference shear rate ascending
    results.sort(key=lambda r: r.metadata["reference_gamma_dot"])
    logger.debug("load_srfs: loaded %d files", len(results))
    return results


def load_series(
    files: list[str | Path] | str,
    protocol: str,
    *,
    sort_by: str | None = None,
    metadata_key: str | None = None,
    metadata_values: list[Any] | None = None,
    format: str | None = None,
    **kwargs: Any,
) -> list[RheoData]:
    """Load a series of files sharing the same rheological protocol.

    A generic multi-file loader that tags each loaded dataset with a protocol
    label and optional metadata, then optionally sorts the resulting list by a
    metadata key.

    Args:
        files: List of file paths or a glob pattern string.
        protocol: Protocol label stored as ``metadata["protocol"]`` on every
            returned dataset (e.g. ``"oscillation"``, ``"relaxation"``).
        sort_by: If provided, sort the output list by ``metadata[sort_by]``
            (ascending). Missing keys are sorted to the end.
        metadata_key: Optional metadata key to tag each dataset with a per-file
            value from *metadata_values*.
        metadata_values: List of values (one per file) written to
            ``metadata[metadata_key]``. Required when *metadata_key* is given.
        format: Optional format hint forwarded to :func:`auto_load`.
        **kwargs: Additional keyword arguments forwarded to :func:`auto_load`.

    Returns:
        List of :class:`~rheojax.core.data.RheoData` objects, optionally sorted.

    Raises:
        FileNotFoundError: If a glob pattern matches no files.
        ValueError: If *metadata_values* length does not match the number of
            files when *metadata_key* is provided.
    """
    paths = _expand_glob(files)

    if metadata_key is not None:
        if metadata_values is None:
            raise ValueError(
                "'metadata_values' must be provided when 'metadata_key' is set."
            )
        if len(metadata_values) != len(paths):
            raise ValueError(
                f"Length of 'metadata_values' ({len(metadata_values)}) does not "
                f"match the number of files ({len(paths)})."
            )

    results: list[RheoData] = []
    for i, path in enumerate(paths):
        logger.debug("load_series: loading file", filepath=str(path), index=i)
        raw = auto_load(path, format=format, **kwargs)
        rd = _flatten_result(raw)
        rd.metadata["protocol"] = protocol
        if metadata_key is not None:
            if metadata_values is None:  # pragma: no cover — guarded above
                raise ValueError("metadata_values required when metadata_key is set")
            rd.metadata[metadata_key] = metadata_values[i]
        results.append(rd)

    if sort_by is not None:
        _sentinel = object()

        def _sort_key(r: RheoData) -> Any:
            val = r.metadata.get(sort_by, _sentinel)
            # Push missing keys to the end by wrapping in a tuple that sorts last
            if val is _sentinel:
                return (1, None)
            return (0, val)

        try:
            results.sort(key=_sort_key)
        except TypeError as exc:
            raise ValueError(
                f"sort_by='{sort_by}' values are not sortable: {exc}"
            ) from exc

    logger.debug("load_series: loaded %d files, protocol='%s'", len(results), protocol)
    return results
