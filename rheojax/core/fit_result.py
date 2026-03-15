"""Structured fit results, model info, and model comparison.

This module provides:
- **FitResult**: Wraps ``OptimizationResult`` with rheology-specific context
  (model name, protocol, parameter units, fitted curve, input data).
- **ModelInfo**: Aggregates ``PluginInfo`` + runtime metadata from a model class.
- **ModelComparison**: Ranks multiple FitResults by information criteria.

FitResult deliberately *delegates* statistical properties (R², AIC, BIC,
confidence intervals, prediction intervals) to the underlying
``OptimizationResult`` — no duplication of computation.

Example:
    >>> model = Maxwell()
    >>> result = model.fit(t, G_data, return_result=True)
    >>> print(result.summary())
    >>> result.plot()
    >>> result.save("maxwell_fit.npz")
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from rheojax.logging import get_logger

if TYPE_CHECKING:
    from rheojax.core.data import RheoData
    from rheojax.utils.optimization import OptimizationResult

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# FitResult
# ---------------------------------------------------------------------------


@dataclass
class FitResult:
    """Structured result from a model fit.

    Wraps an ``OptimizationResult`` and adds rheology-specific context so
    that results can be inspected, serialized, and compared without reaching
    back into the model instance.

    Attributes:
        model_name: Registry name of the model (e.g. ``"maxwell"``).
        model_class_name: Python class name (e.g. ``"Maxwell"``).
        protocol: Test mode / protocol string (e.g. ``"relaxation"``).
        params: Mapping of parameter name → fitted value.
        params_units: Mapping of parameter name → unit string.
        n_params: Number of fitted parameters.
        optimization_result: The underlying ``OptimizationResult``.
        fitted_curve: Model prediction at the training points.
        input_data: Reference to the ``RheoData`` used for fitting (optional).
        X: Raw input array used for fitting.
        y: Raw target array used for fitting.
        timestamp: ISO-8601 timestamp of when the fit was performed.
        metadata: Arbitrary extra info (deformation_mode, poisson_ratio, …).
    """

    model_name: str
    model_class_name: str
    protocol: str | None
    params: dict[str, float]
    params_units: dict[str, str]
    n_params: int
    optimization_result: OptimizationResult | None
    fitted_curve: np.ndarray | None = None
    input_data: RheoData | None = field(default=None, repr=False)
    X: np.ndarray | None = field(default=None, repr=False)
    y: np.ndarray | None = field(default=None, repr=False)
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)
    _fitted_model: Any = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Delegated statistical properties
    # ------------------------------------------------------------------

    @property
    def r_squared(self) -> float | None:
        """Coefficient of determination (R²), delegated to OptimizationResult."""
        if self.optimization_result is None:
            return None
        return self.optimization_result.r_squared

    @property
    def adj_r_squared(self) -> float | None:
        """Adjusted R², delegated to OptimizationResult."""
        if self.optimization_result is None:
            return None
        return self.optimization_result.adj_r_squared

    @property
    def rmse(self) -> float | None:
        """Root mean squared error, delegated to OptimizationResult."""
        if self.optimization_result is None:
            return None
        return self.optimization_result.rmse

    @property
    def mae(self) -> float | None:
        """Mean absolute error, delegated to OptimizationResult."""
        if self.optimization_result is None:
            return None
        return self.optimization_result.mae

    @property
    def aic(self) -> float | None:
        """Akaike Information Criterion, delegated to OptimizationResult."""
        if self.optimization_result is None:
            return None
        return self.optimization_result.aic

    @property
    def bic(self) -> float | None:
        """Bayesian Information Criterion, delegated to OptimizationResult."""
        if self.optimization_result is None:
            return None
        return self.optimization_result.bic

    @property
    def success(self) -> bool:
        """Whether the optimizer converged successfully."""
        if self.optimization_result is None:
            return False
        return self.optimization_result.success

    @property
    def n_data(self) -> int:
        """Number of data points used in the fit."""
        if self.optimization_result is not None:
            return self.optimization_result._resolve_n_data()
        if self.y is not None:
            return len(self.y)
        return 0

    @property
    def converged(self) -> bool:
        """Alias for :attr:`success` (spec compatibility)."""
        return self.success

    @property
    def n_points(self) -> int:
        """Alias for :attr:`n_data` (spec compatibility)."""
        return self.n_data

    @property
    def residuals(self) -> np.ndarray | None:
        """Residual vector from the fit."""
        if self.optimization_result is None:
            return None
        return getattr(self.optimization_result, "residuals", None)

    @property
    def loss_value(self) -> float | None:
        """Final loss (sum of squared residuals)."""
        if self.optimization_result is None:
            return None
        return getattr(self.optimization_result, "fun", None)

    @property
    def n_iterations(self) -> int | None:
        """Number of optimizer iterations."""
        if self.optimization_result is None:
            return None
        return getattr(self.optimization_result, "nit", None)

    @property
    def optimizer_used(self) -> str | None:
        """Name of the optimizer that produced this result."""
        if self.optimization_result is None:
            return self.metadata.get("optimizer_used")
        return getattr(self.optimization_result, "method", None)

    @property
    def covariance(self) -> np.ndarray | None:
        """Parameter covariance matrix from the Jacobian."""
        if self.optimization_result is None:
            return None
        return getattr(self.optimization_result, "pcov", None)

    @property
    def params_ci(self) -> dict[str, tuple[float, float]] | None:
        """Parameter confidence intervals (95%) as {name: (lower, upper)}."""
        ci_arr = self.confidence_intervals(alpha=0.95)
        if ci_arr is None:
            return None
        names = list(self.params.keys())
        return {
            names[i]: (float(ci_arr[i, 0]), float(ci_arr[i, 1]))
            for i in range(min(len(names), len(ci_arr)))
        }

    # ------------------------------------------------------------------
    # Delegated methods
    # ------------------------------------------------------------------

    def confidence_intervals(self, alpha: float = 0.95) -> np.ndarray | None:
        """Parameter confidence intervals from the covariance matrix.

        Args:
            alpha: Confidence level (default 0.95).

        Returns:
            Array of shape ``(n_params, 2)`` or ``None``.
        """
        if self.optimization_result is None:
            return None
        return self.optimization_result.confidence_intervals(alpha)

    def prediction_interval(
        self, x_new: np.ndarray | None = None, alpha: float = 0.95
    ) -> np.ndarray | None:
        """Prediction intervals for new x values.

        Args:
            x_new: New input array (or None for training points).
            alpha: Confidence level (default 0.95).

        Returns:
            Array of shape ``(n_points, 2)`` or ``None``.
        """
        if self.optimization_result is None:
            return None
        return self.optimization_result.prediction_interval(x_new, alpha)

    # ------------------------------------------------------------------
    # New methods
    # ------------------------------------------------------------------

    @property
    def aicc(self) -> float | None:
        """Corrected AIC (AICc) for small samples.

        AICc = AIC + 2k(k+1) / (n - k - 1)
        """
        aic = self.aic
        if aic is None:
            return None
        n = self.n_data
        k = self.n_params
        if n - k - 1 <= 0:
            return np.nan
        return float(aic + 2 * k * (k + 1) / (n - k - 1))

    def summary(self) -> str:
        """Human-readable summary of the fit result.

        Returns:
            Multi-line string with model name, parameters, and statistics.
        """
        lines = [
            f"FitResult: {self.model_class_name} ({self.model_name})",
            f"  Protocol: {self.protocol or 'unknown'}",
            f"  Converged: {self.success}",
            f"  Parameters ({self.n_params}):",
        ]
        ci = self.confidence_intervals()
        for i, (name, value) in enumerate(self.params.items()):
            unit = self.params_units.get(name, "")
            unit_str = f" {unit}" if unit else ""
            ci_str = ""
            if ci is not None:
                ci_str = f"  CI: [{ci[i, 0]:.4g}, {ci[i, 1]:.4g}]"
            lines.append(f"    {name} = {value:.6g}{unit_str}{ci_str}")

        lines.append("  Statistics:")
        for attr_name, label in [
            ("r_squared", "R²"),
            ("adj_r_squared", "Adj R²"),
            ("rmse", "RMSE"),
            ("aic", "AIC"),
            ("bic", "BIC"),
            ("aicc", "AICc"),
        ]:
            val = getattr(self, attr_name)
            if val is not None:
                lines.append(f"    {label}: {val:.6g}")

        lines.append(f"  Timestamp: {self.timestamp}")
        return "\n".join(lines)

    def to_latex(self) -> str:
        """LaTeX table row for this fit result.

        Returns:
            LaTeX-formatted string suitable for inclusion in a tabular environment.
        """
        cols = [self.model_class_name]
        cols.append(str(self.n_params))
        for attr in ("r_squared", "aic", "bic"):
            val = getattr(self, attr)
            cols.append(f"{val:.4f}" if val is not None else "--")
        return " & ".join(cols) + r" \\"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary.

        Returns:
            Dictionary with all scalar fields and numpy arrays as lists.
        """
        result: dict[str, Any] = {
            "model_name": self.model_name,
            "model_class_name": self.model_class_name,
            "protocol": self.protocol,
            "params": dict(self.params),
            "params_units": dict(self.params_units),
            "n_params": self.n_params,
            "r_squared": self.r_squared,
            "adj_r_squared": self.adj_r_squared,
            "rmse": self.rmse,
            "aic": self.aic,
            "bic": self.bic,
            "aicc": self.aicc,
            "success": self.success,
            "n_data": self.n_data,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }
        if self.X is not None:
            result["X"] = np.asarray(self.X).tolist()
        if self.y is not None:
            y_arr = np.asarray(self.y)
            if np.iscomplexobj(y_arr):
                result["y_real"] = y_arr.real.tolist()
                result["y_imag"] = y_arr.imag.tolist()
                result["y_is_complex"] = True
            else:
                result["y"] = y_arr.tolist()
        if self.fitted_curve is not None:
            fc_arr = np.asarray(self.fitted_curve)
            if np.iscomplexobj(fc_arr):
                result["fitted_curve_real"] = fc_arr.real.tolist()
                result["fitted_curve_imag"] = fc_arr.imag.tolist()
                result["fitted_curve_is_complex"] = True
            else:
                result["fitted_curve"] = fc_arr.tolist()
        return result

    def save(self, path: str) -> None:
        """Save fit result to file. Format is dispatched by extension.

        Supported extensions: ``.npz``, ``.json``, ``.h5`` / ``.hdf5``.

        Args:
            path: Output file path.
        """
        from pathlib import Path as _Path

        ext = _Path(path).suffix.lower()
        if ext == ".json":
            self._save_json(path)
        elif ext == ".npz":
            self._save_npz(path)
        elif ext in (".h5", ".hdf5"):
            from rheojax.io.writers.hdf5_writer import save_fit_result_hdf5

            save_fit_result_hdf5(self, path)
        else:
            raise ValueError(
                f"Unsupported extension '{ext}'. Use .json, .npz, .h5, or .hdf5."
            )

    def _save_json(self, path: str) -> None:
        import json

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def _save_npz(self, path: str) -> None:
        import json as _json

        def _str_to_bytes(s: str) -> np.ndarray:
            return np.frombuffer(s.encode("utf-8"), dtype=np.uint8)

        save_dict: dict[str, Any] = {
            "model_name": _str_to_bytes(self.model_name),
            "model_class_name": _str_to_bytes(self.model_class_name),
            "protocol": _str_to_bytes(self.protocol or ""),
            "n_params": np.array(self.n_params),
            "timestamp": _str_to_bytes(self.timestamp),
            "param_names": _str_to_bytes(_json.dumps(list(self.params.keys()))),
            "param_values": np.array(list(self.params.values()), dtype=np.float64),
            # P2-Fit-8: Persist success, n_data, and _is_complex_split for
            # correct round-trip of statistics.
            "success": np.array(self.success),
            "n_data": np.array(self.n_data),
        }
        if self.optimization_result is not None:
            save_dict["_is_complex_split"] = np.array(
                self.optimization_result._is_complex_split
            )
        if self.X is not None:
            save_dict["X"] = np.asarray(self.X)
        if self.y is not None:
            save_dict["y"] = np.asarray(self.y)
        if self.fitted_curve is not None:
            save_dict["fitted_curve"] = np.asarray(self.fitted_curve)
        np.savez(path, **save_dict)

    @classmethod
    def load(cls, path: str) -> FitResult:
        """Load a fit result from a file.

        Args:
            path: Path to a ``.json`` or ``.npz`` file.

        Returns:
            Reconstructed FitResult (without the OptimizationResult reference).
        """
        from pathlib import Path as _Path

        ext = _Path(path).suffix.lower()
        if ext == ".json":
            return cls._load_json(path)
        elif ext == ".npz":
            return cls._load_npz(path)
        else:
            raise ValueError(f"Unsupported extension '{ext}'. Use .json or .npz.")

    @classmethod
    def _load_json(cls, path: str) -> FitResult:
        import json

        with open(path) as f:
            d = json.load(f)

        # Reconstruct arrays, handling complex data serialized as
        # separate real/imag keys (see to_dict).
        def _load_array(
            d: dict,
            key: str,
        ) -> np.ndarray | None:
            if d.get(f"{key}_is_complex"):
                return np.array(d[f"{key}_real"]) + 1j * np.array(d[f"{key}_imag"])
            if key in d:
                return np.array(d[key])
            return None

        y_arr = _load_array(d, "y")
        fitted = _load_array(d, "fitted_curve")

        # Reconstruct a minimal OptimizationResult from serialized stats.
        opt_result = None
        if y_arr is not None and fitted is not None:
            from rheojax.utils.optimization import OptimizationResult

            _is_complex = np.iscomplexobj(y_arr)
            if _is_complex:
                residuals = np.concatenate(
                    [
                        y_arr.real - fitted.real,
                        y_arr.imag - fitted.imag,
                    ]
                )
            else:
                residuals = (y_arr - fitted).ravel()
            rss = float(np.sum(residuals**2))
            opt_result = OptimizationResult(
                x=np.array(list(d["params"].values())),
                fun=rss,
                success=d.get("success", False),
                y_data=y_arr,
                residuals=residuals,
                n_data=len(y_arr),
                _is_complex_split=_is_complex,
            )

        return cls(
            model_name=d["model_name"],
            model_class_name=d["model_class_name"],
            protocol=d.get("protocol"),
            params=d["params"],
            params_units=d.get("params_units", {}),
            n_params=d["n_params"],
            optimization_result=opt_result,
            fitted_curve=fitted,
            X=np.array(d["X"]) if "X" in d else None,
            y=y_arr,
            timestamp=d.get("timestamp", ""),
            metadata=d.get("metadata", {}),
        )

    @classmethod
    def _load_npz(cls, path: str) -> FitResult:
        import json as _json

        def _bytes_to_str(arr: np.ndarray) -> str:
            return arr.tobytes().decode("utf-8")

        data = np.load(path, allow_pickle=False)
        param_names = _json.loads(_bytes_to_str(data["param_names"]))
        param_values = list(data["param_values"])

        y_arr = data["y"] if "y" in data else None
        fitted = data["fitted_curve"] if "fitted_curve" in data else None

        # P2-Fit-8: Read back persisted success, n_data, _is_complex_split
        # with backward-compatible defaults.
        _success = bool(data["success"]) if "success" in data else True
        _n_data = (
            int(data["n_data"])
            if "n_data" in data
            else (len(y_arr) if y_arr is not None else 0)
        )
        _is_complex_split = (
            bool(data["_is_complex_split"]) if "_is_complex_split" in data else False
        )

        # Reconstruct OptimizationResult from y + fitted_curve
        opt_result = None
        if y_arr is not None and fitted is not None:
            from rheojax.utils.optimization import OptimizationResult

            y_np = np.asarray(y_arr)
            fitted_np = np.asarray(fitted)
            if np.iscomplexobj(y_np):
                residuals = np.concatenate(
                    [y_np.real - fitted_np.real, y_np.imag - fitted_np.imag]
                )
            else:
                residuals = (y_np - fitted_np).ravel()
            opt_result = OptimizationResult(
                x=np.array(param_values),
                fun=float(np.sum(residuals**2)),
                success=_success,
                y_data=y_np,
                residuals=residuals,
                n_data=_n_data,
                _is_complex_split=_is_complex_split,
            )

        return cls(
            model_name=_bytes_to_str(data["model_name"]),
            model_class_name=_bytes_to_str(data["model_class_name"]),
            protocol=_bytes_to_str(data["protocol"]) or None,
            params=dict(zip(param_names, param_values, strict=False)),
            params_units={},
            n_params=int(data["n_params"]),
            optimization_result=opt_result,
            fitted_curve=fitted,
            X=data["X"] if "X" in data else None,
            y=y_arr,
            timestamp=_bytes_to_str(data["timestamp"]),
        )

    def plot(self, ax=None, show_residuals: bool = True, **kwargs):
        """Plot the fit result (2-panel: fit + residuals).

        Args:
            ax: Optional matplotlib axes (creates figure if None).
            show_residuals: Whether to show residual panel.
            **kwargs: Passed to matplotlib plot().

        Returns:
            matplotlib Figure.
        """
        import matplotlib.pyplot as plt

        if self.X is None or self.y is None:
            raise ValueError("Cannot plot: X and y data not available.")

        X = np.asarray(self.X)
        y = np.asarray(self.y)

        if show_residuals and self.fitted_curve is not None:
            fig, (ax_fit, ax_res) = plt.subplots(
                2, 1, figsize=(8, 6), height_ratios=[3, 1], sharex=True
            )
        else:
            fig, ax_fit = plt.subplots(1, 1, figsize=(8, 4))
            ax_res = None

        # Handle complex y data
        if np.iscomplexobj(y):
            ax_fit.loglog(X, y.real, "o", label="G' (data)", ms=4, alpha=0.7, **kwargs)
            ax_fit.loglog(X, y.imag, "s", label="G'' (data)", ms=4, alpha=0.7, **kwargs)
            if self.fitted_curve is not None:
                fc = np.asarray(self.fitted_curve)
                if np.iscomplexobj(fc):
                    ax_fit.loglog(X, fc.real, "-", label="G' (fit)", lw=2)
                    ax_fit.loglog(X, fc.imag, "--", label="G'' (fit)", lw=2)
        else:
            ax_fit.plot(X, y, "o", label="Data", ms=4, alpha=0.7, **kwargs)
            if self.fitted_curve is not None:
                ax_fit.plot(X, np.asarray(self.fitted_curve), "-", label="Fit", lw=2)

        ax_fit.set_ylabel("Response")
        ax_fit.legend()
        ax_fit.set_title(
            f"{self.model_class_name} — R² = {self.r_squared:.4f}"
            if self.r_squared is not None
            else self.model_class_name
        )

        if ax_res is not None and self.fitted_curve is not None:
            residuals = y - np.asarray(self.fitted_curve)
            if np.iscomplexobj(residuals):
                ax_res.plot(X, residuals.real, "o", ms=3, label="real")
                ax_res.plot(X, residuals.imag, "s", ms=3, label="imag")
            else:
                ax_res.plot(X, residuals, "o", ms=3)
            ax_res.axhline(0, color="k", lw=0.5)
            ax_res.set_ylabel("Residuals")
            ax_res.set_xlabel("X")

        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# ModelInfo
# ---------------------------------------------------------------------------


@dataclass
class ModelInfo:
    """Aggregated information about a registered model.

    Combines ``PluginInfo`` from the registry with runtime metadata obtained
    by temporarily instantiating the model class.

    Attributes:
        name: Registry name (e.g. ``"maxwell"``).
        class_name: Python class name (e.g. ``"Maxwell"``).
        param_names: List of parameter names.
        param_bounds: Mapping of name → (lower, upper).
        param_units: Mapping of name → unit string.
        n_params: Number of parameters.
        protocols: List of supported Protocol enum values.
        deformation_modes: List of supported DeformationMode enum values.
        supports_bayesian: Whether the model supports Bayesian inference.
        doc: First paragraph of the model docstring.
    """

    name: str
    class_name: str
    model_class: type | None
    param_names: list[str]
    param_bounds: dict[str, tuple[float, float]]
    param_units: dict[str, str]
    n_params: int
    protocols: list[Any]
    deformation_modes: list[Any]
    supports_bayesian: bool
    doc: str | None = None

    @classmethod
    def from_registry(cls, name: str) -> ModelInfo:
        """Construct ModelInfo by inspecting a registered model.

        Temporarily instantiates the model class to read parameter metadata.

        Args:
            name: Registry name of the model.

        Returns:
            Populated ModelInfo instance.

        Raises:
            KeyError: If the model is not registered.
        """
        from rheojax.core.registry import ModelRegistry

        info = ModelRegistry.get_info(name)
        if info is None:
            raise KeyError(f"Model '{name}' not found in registry.")

        # Instantiate to read parameter metadata
        try:
            instance = info.plugin_class()
            param_names = list(instance.parameters.keys())
            param_bounds = {}
            param_units = {}
            for pname in param_names:
                p = instance.parameters[pname]
                param_bounds[pname] = p.bounds if p.bounds else (None, None)
                param_units[pname] = getattr(p, "units", "") or ""
            n_params = len(param_names)
            supports_bayesian = hasattr(instance, "fit_bayesian")
        except Exception:
            logger.warning(
                "ModelInfo.from_registry: instantiation failed for model '%s'",
                name,
                exc_info=True,
            )
            param_names = []
            param_bounds = {}
            param_units = {}
            n_params = 0
            supports_bayesian = False

        return cls(
            name=name,
            class_name=info.plugin_class.__name__,
            model_class=info.plugin_class,
            param_names=param_names,
            param_bounds=param_bounds,
            param_units=param_units,
            n_params=n_params,
            protocols=list(info.protocols),
            deformation_modes=list(info.deformation_modes),
            supports_bayesian=supports_bayesian,
            doc=info.doc,
        )


# ---------------------------------------------------------------------------
# ModelComparison
# ---------------------------------------------------------------------------


@dataclass
class ModelComparison:
    """Comparison of multiple model fit results.

    Ranks models by an information criterion (AIC, BIC, or AICc) and
    computes Akaike weights for model-averaging.

    Attributes:
        results: List of FitResult objects.
        criterion: Which criterion was used for ranking (``"aic"``, ``"bic"``, ``"aicc"``).
        rankings: Model names sorted from best (index 0) to worst.
        delta_criterion: Δ_i = criterion_i - criterion_best.
        weights: Akaike weights w_i = exp(-0.5 Δ_i) / Σ exp(-0.5 Δ_j).
        best_model: Name of the best model.
    """

    results: list[FitResult]
    criterion: str = "aic"
    rankings: dict[str, int] = field(default_factory=dict)
    delta_criterion: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    best_model: str = ""

    _VALID_CRITERIA = {"aic", "bic", "aicc"}

    def __post_init__(self):
        if self.criterion not in self._VALID_CRITERIA:
            raise ValueError(
                f"criterion must be one of {self._VALID_CRITERIA!r}, "
                f"got {self.criterion!r}"
            )
        if self.results and not self.rankings:
            self._compute_rankings()

    def _compute_rankings(self) -> None:
        """Compute rankings and Akaike weights from results."""
        # Collect criterion values
        entries: list[tuple[str, float]] = []
        for r in self.results:
            val = getattr(r, self.criterion, None)
            if val is not None and np.isfinite(val):
                entries.append((r.model_name, float(val)))

        if not entries:
            return

        # Sort by criterion (lower is better)
        entries.sort(key=lambda e: e[1])
        best_val = entries[0][1]

        self.rankings = {name: rank for rank, (name, _) in enumerate(entries, 1)}
        self.best_model = entries[0][0]

        # Delta and Akaike weights
        deltas = {name: val - best_val for name, val in entries}
        self.delta_criterion = deltas

        raw_weights = {name: np.exp(-0.5 * d) for name, d in deltas.items()}
        total = sum(raw_weights.values())
        if total > 0:
            self.weights = {name: w / total for name, w in raw_weights.items()}
        else:
            self.weights = dict.fromkeys(deltas, 0.0)

    def ranked_names(self) -> list[str]:
        """Return model names sorted by rank (best first)."""
        return sorted(self.rankings, key=lambda n: self.rankings[n])

    def summary(self) -> str:
        """Human-readable summary table.

        Returns:
            Formatted string with model rankings.
        """
        lines = [
            f"Model Comparison (criterion: {self.criterion.upper()})",
            f"{'Rank':<6}{'Model':<25}{'Criterion':<14}{'Delta':<12}{'Weight':<10}",
            "-" * 67,
        ]
        for name in self.ranked_names():
            rank = self.rankings[name]
            r = next((x for x in self.results if x.model_name == name), None)
            crit_val = getattr(r, self.criterion, None) if r else None
            delta = self.delta_criterion.get(name, float("nan"))
            weight = self.weights.get(name, 0.0)
            crit_str = f"{crit_val:.4f}" if crit_val is not None else "--"
            lines.append(
                f"{rank:<6}{name:<25}{crit_str:<14}{delta:<12.4f}{weight:<10.4f}"
            )
        return "\n".join(lines)

    def plot(self, ax=None, **kwargs):
        """Bar plot of Akaike weights.

        Args:
            ax: Optional matplotlib axes.
            **kwargs: Passed to matplotlib bar().

        Returns:
            matplotlib Figure.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            fig = ax.figure

        names = self.ranked_names()
        w = [self.weights.get(n, 0.0) for n in names]
        ax.bar(range(len(names)), w, tick_label=names, **kwargs)
        ax.set_ylabel("Akaike Weight")
        ax.set_title(f"Model Comparison ({self.criterion.upper()})")
        ax.set_ylim(0, 1.05)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha("right")
        fig.tight_layout()
        return fig
