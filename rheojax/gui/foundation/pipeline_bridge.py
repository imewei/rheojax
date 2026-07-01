from __future__ import annotations

from rheojax.gui.foundation.library import DatasetLibrary, DatasetRef


def pipeline_inputs_from_library(lib: DatasetLibrary, ids: list[str]) -> list[DatasetRef]:
    """§11 boundary: feed the legacy pipeline from the new Dataset Library.

    Resolves *ids* to ``DatasetRef`` objects in the order given.
    Raises ``KeyError`` for any id not present in *lib*.
    Returns an empty list when *ids* is empty.
    """
    return [lib.get(i) for i in ids]
