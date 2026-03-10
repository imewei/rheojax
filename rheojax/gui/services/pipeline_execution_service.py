"""Pipeline execution service for GUI.

Orchestrates sequential execution of visual pipeline steps through
existing GUI services. Each step type delegates to the appropriate
service (DataService, TransformService, ModelService, etc.).

This service runs SYNCHRONOUSLY — the caller is responsible for
invoking it from a worker thread, not the GUI thread.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal

from rheojax.gui.state.store import StateStore, StepStatus
from rheojax.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# DMTA parameter resolution helper (P1-1, P2-1, P2-4)
# ---------------------------------------------------------------------------


def _resolve_dmta_kwargs(kwargs: dict[str, Any], data: Any) -> None:
    """Resolve deformation_mode and poisson_ratio into *kwargs* in-place.

    Fallback chain (first non-None wins):
      1. Already present in *kwargs* (from step config).
      2. data.metadata (set by data loader / prior pipeline steps).
      3. AppState global settings (user's UI selection).
    """
    _meta: dict = {}
    if hasattr(data, "metadata") and data.metadata:
        _meta = data.metadata

    if "deformation_mode" not in kwargs:
        dm = _meta.get("deformation_mode")
        if dm is None:
            try:
                store = StateStore()
                dm = getattr(store.get_state(), "deformation_mode", None)
            except Exception:
                pass
        if dm is not None:
            kwargs["deformation_mode"] = dm

    if "poisson_ratio" not in kwargs:
        pr = _meta.get("poisson_ratio")
        if pr is None:
            try:
                store = StateStore()
                pr = getattr(store.get_state(), "poisson_ratio", None)
            except Exception:
                pass
        if pr is not None:
            kwargs["poisson_ratio"] = pr


# ---------------------------------------------------------------------------
# Internal state-mutation helpers
# (update_step_status, cache_step_result, set_pipeline_running do not exist
# in actions.py, so we implement them here via direct StateStore updaters.)
# ---------------------------------------------------------------------------


def _update_step_status(
    step_id: str,
    status: StepStatus,
    error_message: str | None = None,
) -> None:
    """Update the status of a PipelineStepConfig in VisualPipelineState."""
    store = StateStore()

    def updater(state):
        vp = state.visual_pipeline.clone()
        new_steps = []
        for step in vp.steps:
            if step.id == step_id:
                updated = replace(step, status=status, error_message=error_message)
                new_steps.append(updated)
            else:
                new_steps.append(step)
        vp.steps = new_steps
        return replace(state, visual_pipeline=vp)

    store.update_state(updater)
    store.emit_signal("visual_pipeline_changed")


def _cache_step_result(step_id: str, result: Any) -> None:
    """Cache the result of a completed step in VisualPipelineState.step_results."""
    store = StateStore()

    def updater(state):
        vp = state.visual_pipeline.clone()
        vp.step_results[step_id] = result
        return replace(state, visual_pipeline=vp)

    store.update_state(updater)


def _set_pipeline_running(
    is_running: bool,
    current_step_id: str | None = None,
) -> None:
    """Set the running flag and active step on VisualPipelineState."""
    store = StateStore()

    def updater(state):
        vp = state.visual_pipeline.clone()
        vp.is_running = is_running
        vp.current_running_step_id = current_step_id
        return replace(state, visual_pipeline=vp)

    store.update_state(updater)
    store.emit_signal("visual_pipeline_changed")


class PipelineExecutionService(QObject):
    """Executes visual pipeline steps sequentially.

    Signals
    -------
    step_started(str):
        Emitted when a step begins execution (carries step_id).
    step_completed(str):
        Emitted when a step finishes successfully (carries step_id).
    step_failed(str, str):
        Emitted when a step raises an exception (step_id, error_message).
    pipeline_started():
        Emitted once before the first step runs.
    pipeline_completed():
        Emitted after all steps complete without error.
    pipeline_failed(str):
        Emitted if any step raises an uncaught exception (error_message).
    """

    step_started = Signal(str)
    step_completed = Signal(str)
    step_failed = Signal(str, str)
    pipeline_started = Signal()
    pipeline_completed = Signal()
    pipeline_failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        # Lazy service creation — instantiated on first use per step type.
        self._data_service = None
        self._transform_service = None
        self._model_service = None
        self._bayesian_service = None
        self._export_service = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_all(self, steps: list) -> None:
        """Execute all pipeline steps sequentially.

        This method runs synchronously.  It must be called from a worker
        thread, not the GUI thread, to avoid blocking the event loop.

        Parameters
        ----------
        steps : list[PipelineStepConfig]
            Ordered list of steps to execute.
        """
        if not steps:
            _set_pipeline_running(False)
            self.pipeline_started.emit()
            self.pipeline_completed.emit()
            return

        _set_pipeline_running(True, steps[0].id)
        self.pipeline_started.emit()

        # Accumulates results shared across steps (data, fit_result, etc.)
        context: dict[str, Any] = {}

        try:
            for step in steps:
                _set_pipeline_running(True, step.id)
                _update_step_status(step.id, StepStatus.ACTIVE)
                self.step_started.emit(step.id)

                try:
                    result = self._execute_step(step, context)
                    _cache_step_result(step.id, result)
                    _update_step_status(step.id, StepStatus.COMPLETE)
                    self.step_completed.emit(step.id)
                except Exception as exc:
                    error_msg = str(exc)
                    _update_step_status(step.id, StepStatus.ERROR, error_msg)
                    self.step_failed.emit(step.id, error_msg)
                    raise

            _set_pipeline_running(False)
            self.pipeline_completed.emit()

        except Exception as exc:
            _set_pipeline_running(False)
            self.pipeline_failed.emit(str(exc))

    def execute_single_step(self, step, context: dict) -> Any:
        """Execute a single step with the given context.

        Wraps the step in the same status-dispatch lifecycle as
        ``execute_all``.  The caller is responsible for providing a
        ``context`` dict that accumulates inter-step results.

        Parameters
        ----------
        step : PipelineStepConfig
            Step to execute.
        context : dict
            Accumulated results from previous steps (mutated in place).

        Returns
        -------
        Any
            The step result.

        Raises
        ------
        Exception
            Re-raises any exception from the underlying service after
            dispatching ERROR status.
        """
        _update_step_status(step.id, StepStatus.ACTIVE)
        self.step_started.emit(step.id)

        try:
            result = self._execute_step(step, context)
            _cache_step_result(step.id, result)
            _update_step_status(step.id, StepStatus.COMPLETE)
            self.step_completed.emit(step.id)
            return result
        except Exception as exc:
            error_msg = str(exc)
            _update_step_status(step.id, StepStatus.ERROR, error_msg)
            self.step_failed.emit(step.id, error_msg)
            raise

    # ------------------------------------------------------------------
    # Internal routing
    # ------------------------------------------------------------------

    def _execute_step(self, step, context: dict) -> Any:
        """Route a step to the appropriate service method."""
        dispatch = {
            "load": self._execute_load,
            "transform": self._execute_transform,
            "fit": self._execute_fit,
            "bayesian": self._execute_bayesian,
            "export": self._execute_export,
        }
        handler = dispatch.get(step.step_type)
        if handler is None:
            raise ValueError(
                f"Unknown step type '{step.step_type}'. "
                f"Expected one of: {list(dispatch)}"
            )
        return handler(step.config, context)

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    def _execute_load(self, config: dict, context: dict) -> Any:
        """Execute a load step via DataService.load_file.

        Recognised config keys
        ----------------------
        file : str
            Path to the data file (required).
        x_col : str, optional
            Column name for the independent variable.
        y_col : str, optional
            Column name for the primary response variable.
        y2_col : str, optional
            Column name for the secondary response (e.g. G'' in oscillation).
        test_mode : str, optional
            Override auto-detected test mode.

        Any unrecognised keys are forwarded to DataService.load_file as
        ``**kwargs``, which passes them through to the underlying reader.
        """
        from rheojax.gui.services.data_service import DataService

        if self._data_service is None:
            self._data_service = DataService()

        file_path = config.get("file", "")
        if not file_path:
            raise ValueError("Load step requires a 'file' key in its config.")

        # Defense-in-depth: reject path-traversal attempts.  Absolute paths are
        # legitimate (GUI file-picker, CLI --output flag), so only '..' is blocked.
        _path = Path(file_path)
        if ".." in _path.parts:
            raise ValueError(
                f"Load step file path must not contain '..' segments: {file_path}"
            )

        # Extract the known DataService.load_file keyword arguments.
        _KNOWN = {"x_col", "y_col", "y2_col", "test_mode"}
        load_kwargs: dict[str, Any] = {
            k: v for k, v in config.items() if k != "file" and k in _KNOWN
        }
        # Forward any extra reader-specific kwargs (e.g. delimiter, sheet_name).
        extra_kwargs: dict[str, Any] = {
            k: v for k, v in config.items() if k != "file" and k not in _KNOWN
        }
        load_kwargs.update(extra_kwargs)

        logger.info(
            "Pipeline load step",
            file_path=str(file_path),
            load_kwargs=list(load_kwargs.keys()),
        )

        data = self._data_service.load_file(str(file_path), **load_kwargs)
        context["data"] = data
        return data

    def _execute_transform(self, config: dict, context: dict) -> Any:
        """Execute a transform step via TransformService.apply_transform.

        Recognised config keys
        ----------------------
        name : str
            Transform name (e.g. "fft", "mastercurve", "derivative").
        All remaining keys are passed as transform parameters.

        Context requirements
        --------------------
        context["data"] must be set by a prior load step.
        """
        from rheojax.gui.services.transform_service import TransformService

        if self._transform_service is None:
            self._transform_service = TransformService()

        data = context.get("data")
        if data is None:
            raise ValueError(
                "Transform step requires data from a prior load step. "
                "No 'data' found in execution context."
            )

        transform_name = config.get("name", "")
        if not transform_name:
            raise ValueError("Transform step requires a 'name' key in its config.")

        params = {k: v for k, v in config.items() if k != "name"}

        logger.info(
            "Pipeline transform step",
            transform=transform_name,
            n_params=len(params),
        )

        result = self._transform_service.apply_transform(transform_name, data, params)

        # apply_transform returns RheoData or (RheoData, extras_dict).
        if isinstance(result, tuple):
            transformed_data = result[0]
        else:
            transformed_data = result

        context["data"] = transformed_data
        return transformed_data

    def _execute_fit(self, config: dict, context: dict) -> Any:
        """Execute a fit step via ModelService.fit.

        Recognised config keys
        ----------------------
        model : str
            Model name (required).
        params : dict, optional
            Initial parameter values. If omitted the model's defaults are used.
        test_mode : str, optional
            Override auto-detected test mode.
        All remaining keys are forwarded as fit_kwargs (e.g. max_iter, ftol).

        Context requirements
        --------------------
        context["data"] must be set by a prior load step.
        """
        from rheojax.gui.services.model_service import ModelService

        if self._model_service is None:
            self._model_service = ModelService()

        data = context.get("data")
        if data is None:
            raise ValueError(
                "Fit step requires data from a prior load step. "
                "No 'data' found in execution context."
            )

        model_name = config.get("model", "")
        if not model_name:
            raise ValueError("Fit step requires a 'model' key in its config.")

        # ModelService.fit signature: fit(model_name, data, params, test_mode, ...)
        # params is a mandatory positional argument (dict of {name: value}).
        initial_params: dict[str, Any] = config.get("params") or {}

        # test_mode can be overridden via config or inferred from data metadata.
        test_mode: str | None = config.get("test_mode")
        if test_mode is None and hasattr(data, "metadata") and data.metadata:
            test_mode = data.metadata.get("test_mode")

        # All other keys that aren't model/params/test_mode are fit kwargs.
        _RESERVED = {"model", "params", "test_mode"}
        fit_kwargs: dict[str, Any] = {
            k: v for k, v in config.items() if k not in _RESERVED
        }

        # P1-1 / P2-4: Resolve deformation_mode and poisson_ratio from
        # (a) step config (already in fit_kwargs), (b) data metadata,
        # (c) AppState global settings — in that priority order.
        _resolve_dmta_kwargs(fit_kwargs, data)

        logger.info(
            "Pipeline fit step",
            model=model_name,
            test_mode=test_mode,
            n_fit_kwargs=len(fit_kwargs),
        )

        fit_result = self._model_service.fit(
            model_name,
            data,
            initial_params,
            test_mode=test_mode,
            **fit_kwargs,
        )

        context["fit_result"] = fit_result
        context["model_name"] = model_name
        return fit_result

    def _execute_bayesian(self, config: dict, context: dict) -> Any:
        """Execute a Bayesian inference step via BayesianService.run_mcmc.

        Recognised config keys
        ----------------------
        num_warmup : int, default=1000
        num_samples : int, default=2000
        num_chains : int, default=4
        warm_start : bool, default=True
            Whether to initialise from the NLSQ fit result.
        All remaining keys are forwarded to run_mcmc as **kwargs
        (e.g. target_accept_prob, custom_priors).

        Context requirements
        --------------------
        context["data"] and context["model_name"] must be present.
        context["fit_result"] is optional but enables warm-starting.
        """
        from rheojax.gui.services.bayesian_service import BayesianService

        if self._bayesian_service is None:
            self._bayesian_service = BayesianService()

        data = context.get("data")
        if data is None:
            raise ValueError(
                "Bayesian step requires data from a prior load step. "
                "No 'data' found in execution context."
            )

        model_name = context.get("model_name")
        if model_name is None:
            raise ValueError(
                "Bayesian step requires a prior fit step. "
                "No 'model_name' found in execution context."
            )

        num_warmup: int = int(config.get("num_warmup", 1000))
        num_samples: int = int(config.get("num_samples", 2000))
        num_chains: int = int(config.get("num_chains", 4))
        use_warm_start: bool = bool(config.get("warm_start", True))

        # Build warm-start param dict from the prior NLSQ fit result.
        warm_start_params: dict[str, float] | None = None
        fitted_model_state: dict | None = None

        fit_result = context.get("fit_result")
        if fit_result is not None and use_warm_start:
            if hasattr(fit_result, "parameters") and fit_result.parameters:
                warm_start_params = dict(fit_result.parameters)
            # Transfer cached model protocol state so run_mcmc can restore it
            # on the fresh model instance (see BayesianService.run_mcmc docs).
            if hasattr(fit_result, "metadata") and isinstance(
                fit_result.metadata, dict
            ):
                fitted_model_state = fit_result.metadata.get("model_state")

        # Collect remaining kwargs (e.g. target_accept_prob, custom_priors).
        _RESERVED = {"num_warmup", "num_samples", "num_chains", "warm_start"}
        extra_kwargs: dict[str, Any] = {
            k: v for k, v in config.items() if k not in _RESERVED
        }
        if fitted_model_state is not None:
            extra_kwargs["fitted_model_state"] = fitted_model_state

        # P2-1: Resolve deformation_mode, poisson_ratio, and test_mode for
        # Bayesian inference — same fallback chain as _execute_fit.
        _resolve_dmta_kwargs(extra_kwargs, data)
        if "test_mode" not in extra_kwargs:
            _tm: str | None = None
            if hasattr(data, "metadata") and data.metadata:
                _tm = data.metadata.get("test_mode")
            if _tm is not None:
                extra_kwargs["test_mode"] = _tm

        logger.info(
            "Pipeline Bayesian step",
            model=model_name,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            warm_start=use_warm_start,
        )

        result = self._bayesian_service.run_mcmc(
            model_name=model_name,
            data=data,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            warm_start=warm_start_params,
            **extra_kwargs,
        )

        context["bayesian_result"] = result
        return result

    def _execute_export(self, config: dict, context: dict) -> Any:
        """Execute an export step via ExportService.

        Recognised config keys
        ----------------------
        output : str, default="results/"
            Output directory path (created if absent).
        format : str, default="directory"
            Export format:
            - "directory" — write data.csv + parameters.csv + posterior.csv
              into *output* using ExportService helpers.
            - "csv", "json", "xlsx", "hdf5" — export fitted parameters only
              (via ExportService.export_parameters).

        Notes
        -----
        AnalysisExporter.export_directory / export_excel both require a
        Pipeline object as their first argument; they cannot be driven from
        loose data/result dicts.  For the GUI pipeline we therefore use
        ExportService directly, which provides per-result export methods
        with the right signatures.
        """
        from rheojax.gui.services.export_service import ExportService

        if self._export_service is None:
            self._export_service = ExportService()

        output_path = Path(config.get("output", "results/"))
        export_format: str = config.get("format", "directory")

        # Defense-in-depth: reject path-traversal attempts only.  Absolute paths
        # are valid for GUI file-picker and CLI --output invocations.
        if ".." in output_path.parts:
            raise ValueError(
                f"Export output path must not contain '..' segments: {output_path}"
            )

        output_path.mkdir(parents=True, exist_ok=True)

        fit_result = context.get("fit_result")
        bayesian_result = context.get("bayesian_result")
        data = context.get("data")

        logger.info(
            "Pipeline export step",
            output=str(output_path),
            format=export_format,
            has_data=data is not None,
            has_fit=fit_result is not None,
            has_bayesian=bayesian_result is not None,
        )

        if export_format == "directory":
            # Write each available artefact into the output directory using
            # ExportService methods that accept individual objects.
            if data is not None:
                try:
                    self._export_service.export_data(
                        data, output_path / "data.csv", format="csv"
                    )
                except Exception as exc:  # pragma: no cover
                    logger.warning("Could not export data CSV", error=str(exc))

            if fit_result is not None:
                try:
                    self._export_service.export_parameters(
                        fit_result, output_path / "parameters.csv", format="csv"
                    )
                except Exception as exc:  # pragma: no cover
                    logger.warning("Could not export fit parameters", error=str(exc))

            if bayesian_result is not None:
                try:
                    self._export_service.export_posterior(
                        bayesian_result,
                        output_path / "posterior.csv",
                        format="csv",
                    )
                except Exception as exc:  # pragma: no cover
                    logger.warning("Could not export posterior samples", error=str(exc))

        else:
            # Single-format parameter export (csv / json / xlsx / hdf5).
            if fit_result is not None:
                ext_map = {
                    "csv": ".csv",
                    "json": ".json",
                    "xlsx": ".xlsx",
                    "hdf5": ".h5",
                }
                ext = ext_map.get(export_format, f".{export_format}")
                param_path = output_path / f"parameters{ext}"
                self._export_service.export_parameters(
                    fit_result, param_path, format=export_format
                )
            elif bayesian_result is not None:
                ext_map = {"csv": ".csv", "xlsx": ".xlsx", "hdf5": ".h5"}
                ext = ext_map.get(export_format, f".{export_format}")
                post_path = output_path / f"posterior{ext}"
                self._export_service.export_posterior(
                    bayesian_result, post_path, format=export_format
                )
            else:
                logger.warning(
                    "Export step: no fit_result or bayesian_result in context; "
                    "nothing to export."
                )

        context["export_path"] = str(output_path)
        return str(output_path)
