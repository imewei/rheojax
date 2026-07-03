from __future__ import annotations

from typing import Any

from rheojax.gui.foundation.library import DatasetLibrary, DatasetRef


def pipeline_inputs_from_library(lib: DatasetLibrary, ids: list[str]) -> list[DatasetRef]:
    """§11 boundary: feed the legacy pipeline from the new Dataset Library.

    Resolves *ids* to ``DatasetRef`` objects in the order given.
    Raises ``KeyError`` for any id not present in *lib*.
    Returns an empty list when *ids* is empty.
    """
    return [lib.get(i) for i in ids]


def pipeline_context_from_library(lib: DatasetLibrary, ids: list[str]) -> dict[str, Any]:
    """Seed a PipelineExecutionService.execute_single_step context from a
    Workspace Dataset Library id, so a pipeline run can start from any
    non-load step (transform/fit/bayesian/export) against Workspace data
    (imported or derived) without PipelineExecutionService._execute_load's
    file-path-only re-read.

    Only the first id is used (a pipeline step's ``context["data"]`` holds a
    single RheoData at a time, per PipelineExecutionService._execute_transform
    /_execute_fit/_execute_bayesian/_execute_export). Returns an empty dict
    for an empty *ids* list (nothing to seed).
    """
    if not ids:
        return {}
    return {"data": lib.load_payload(ids[0]), "dataset_id": ids[0]}
