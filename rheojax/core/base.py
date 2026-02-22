"""Base classes for models and transforms with JAX support.

This module provides abstract base classes that define consistent interfaces
for all models and transforms in the rheojax package, with full JAX support.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from rheojax.core.bayesian import BayesianMixin, BayesianResult
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.parameters import Parameter, ParameterSet
from rheojax.core.test_modes import DeformationMode
from rheojax.logging import get_logger

# Module-level logger
logger = get_logger(__name__)

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()

# Type alias for arrays (accepts both NumPy and JAX arrays)
# Note: jnp.ndarray is dynamically imported, so we use np.ndarray for type checking
type ArrayLike = np.ndarray


class BaseModel(BayesianMixin, ABC):
    """Abstract base class for all rheological models.

    This class defines the standard interface that all models must implement,
    supporting JAX arrays, scikit-learn style APIs,
    and Bayesian inference via NumPyro NUTS.

    All models inherit Bayesian capabilities from BayesianMixin, including:
    - fit_bayesian(): Bayesian parameter estimation using NUTS
    - sample_prior(): Sample from prior distributions
    - get_credible_intervals(): Compute highest density intervals

    The fit() method uses NLSQ optimization by default for fast point estimation,
    which can be used to warm-start Bayesian inference.
    """

    def __init__(self):
        """Initialize base model."""
        logger.debug("Initializing model", model=self.__class__.__name__)
        self.parameters = ParameterSet()
        self.fitted_ = False
        self._nlsq_result = None  # Store NLSQ optimization result
        self._bayesian_result = None  # Store Bayesian inference result
        self.X_data = None  # Store data for Bayesian inference
        self.y_data = None
        self._deformation_mode: DeformationMode | None = None
        self._poisson_ratio: float = 0.5

    @abstractmethod
    def _fit(self, X: ArrayLike, y: ArrayLike, **kwargs) -> BaseModel:
        """Internal fit implementation to be overridden by subclasses.

        Args:
            X: Input features
            y: Target values
            **kwargs: Additional fitting options

        Returns:
            self for method chaining
        """
        pass

    @abstractmethod
    def _predict(self, X: ArrayLike, **kwargs) -> ArrayLike:
        """Internal predict implementation to be overridden by subclasses.

        Args:
            X: Input features
            **kwargs: Additional prediction options

        Returns:
            Predictions
        """
        pass

    def _detect_optimization_strategy(
        self,
        X: ArrayLike,
        use_log_residuals: bool | None,
        use_multi_start: bool | None,
        n_starts: int,
    ) -> tuple[bool, bool]:
        """Auto-detect optimization strategy based on data range.

        Args:
            X: Input data
            use_log_residuals: User-specified setting or None for auto-detect
            use_multi_start: User-specified setting or None for auto-detect
            n_starts: Number of starts for multi-start optimization

        Returns:
            Tuple of (use_log_residuals, use_multi_start) with defaults applied
        """
        if use_log_residuals is not None and use_multi_start is not None:
            return use_log_residuals, use_multi_start

        try:
            from rheojax.core.data import RheoData
            from rheojax.utils.data_quality import detect_data_range_decades

            x_array = X.x if isinstance(X, RheoData) else X
            decades = detect_data_range_decades(x_array)

            if use_log_residuals is None:
                if decades > 8.0:
                    use_log_residuals = True
                    logger.info(
                        "Auto-enabling log-residuals for wide range",
                        model=self.__class__.__name__,
                        decades=f"{decades:.1f}",
                    )
                else:
                    use_log_residuals = False

            if use_multi_start is None:
                if decades > 10.0:
                    use_multi_start = True
                    logger.info(
                        "Auto-enabling multi-start optimization for very wide range",
                        model=self.__class__.__name__,
                        decades=f"{decades:.1f}",
                        n_starts=n_starts,
                    )
                else:
                    use_multi_start = False

        except Exception as e:
            logger.debug(
                "Range detection failed",
                model=self.__class__.__name__,
                error=str(e),
            )
            use_log_residuals = (
                use_log_residuals if use_log_residuals is not None else False
            )
            use_multi_start = use_multi_start if use_multi_start is not None else False

        return use_log_residuals, use_multi_start

    def _check_compatibility(
        self,
        X: ArrayLike,
        y: ArrayLike,
        test_mode: str | None,
    ) -> dict | None:
        """Check model-data compatibility and return result.

        Args:
            X: Input data
            y: Target data
            test_mode: Test mode ('relaxation', 'oscillation', etc.)

        Returns:
            Compatibility dict if check succeeds, None otherwise
        """
        try:
            from rheojax.utils.compatibility import check_model_compatibility

            return check_model_compatibility(
                model=self,
                t=X if test_mode == "relaxation" else None,
                G_t=y if test_mode == "relaxation" else None,
                omega=X if test_mode == "oscillation" else None,
                G_star=y if test_mode == "oscillation" else None,
                test_mode=test_mode,
            )
        except Exception as exc:
            logger.debug(
                "Compatibility check failed",
                model=self.__class__.__name__,
                error=str(exc),
            )
            return None

    def _enhance_error_with_compatibility(
        self,
        error: RuntimeError,
        X: ArrayLike,
        y: ArrayLike,
        test_mode: str | None,
    ) -> RuntimeError:
        """Enhance optimization error with compatibility information.

        Args:
            error: Original RuntimeError
            X: Input data
            y: Target data
            test_mode: Test mode

        Returns:
            Enhanced RuntimeError or original if enhancement fails
        """
        error_msg = str(error)

        if (
            "Optimization failed" not in error_msg
            and "did not converge" not in error_msg
        ):
            return error

        compatibility = self._check_compatibility(X, y, test_mode)
        if compatibility is None or compatibility.get("compatible", True):
            return error

        try:
            from rheojax.utils.compatibility import format_compatibility_message

            compat_msg = format_compatibility_message(compatibility)
            enhanced_msg = (
                f"{error_msg}\n\n"
                f"Model-data compatibility issue detected:\n"
                f"{compat_msg}\n\n"
                f"Note: This model may not be appropriate for your data. "
                f"In model comparison pipelines, it's normal for some models "
                f"to fail when their underlying physics doesn't match the material behavior."
            )
            return RuntimeError(enhanced_msg)
        except Exception as exc:
            logger.debug(
                "Failed to enhance error with compatibility info",
                error=str(exc),
            )
            return error

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        method: str = "nlsq",
        check_compatibility: bool = False,
        use_log_residuals: bool | None = None,
        use_multi_start: bool | None = None,
        n_starts: int = 5,
        perturb_factor: float = 0.3,
        deformation_mode: str | DeformationMode | None = None,
        poisson_ratio: float = 0.5,
        **kwargs,
    ) -> BaseModel:
        """Fit the model to data using NLSQ optimization.

        This method uses NLSQ (GPU-accelerated nonlinear least squares) by default
        for fast point estimation. The optimization result is stored for potential
        warm-starting of Bayesian inference.

        For very wide frequency ranges (>10 decades), multi-start optimization is
        automatically enabled to escape local minima.

        Args:
            X: Input features
            y: Target values
            method: Optimization method ('nlsq' by default for compatibility)
            check_compatibility: Whether to check model-data compatibility before
                fitting. If True, warns when model may not be appropriate for data.
                Default is False for backward compatibility.
            use_log_residuals: Whether to use log-space residuals for fitting.
                Recommended for wide frequency ranges (>8 decades) to prevent
                optimizer bias. If None (default), automatically detected based
                on data range. Explicit True/False overrides auto-detection.
            use_multi_start: Whether to use multi-start optimization to escape
                local minima. Recommended for very wide ranges (>10 decades).
                If None (default), automatically enabled for >10 decades.
            n_starts: Number of random starts for multi-start optimization (default: 5)
            perturb_factor: Perturbation magnitude for multi-start random starts (default: 0.3).
                Parameters are perturbed by ± perturb_factor * (value or range).
                Larger values (0.7-0.9) explore wider parameter space.
            **kwargs: Additional fitting options passed to _fit()

        Returns:
            self for method chaining (scikit-learn style)

        Example:
            >>> model = Maxwell()
            >>> model.fit(t, G_data)  # Uses NLSQ by default
            >>> model.fit(t, G_data, method='nlsq', max_iter=1000)
            >>> model.fit(t, G_data, check_compatibility=True)  # Check compatibility
            >>> model.fit(omega, G_star, use_log_residuals=True)  # Force log-residuals
            >>> model.fit(mastercurve, None, use_multi_start=True, n_starts=10)  # Multi-start
        """
        # Get data shape for logging
        data_shape = getattr(X, "shape", None) or (len(X),)
        logger.debug(
            "Entering fit",
            model=self.__class__.__name__,
            data_shape=data_shape,
            method=method,
        )

        # Handle deformation mode: auto-detect from RheoData or explicit parameter
        if deformation_mode is None:
            from rheojax.core.data import RheoData

            if isinstance(X, RheoData):
                deformation_mode = X.metadata.get("deformation_mode", None)
                # When auto-detecting tensile mode from RheoData, extract x/y
                # so that downstream _fit() uses the converted y, not RheoData.y
                if deformation_mode is not None:
                    if y is None:
                        y = X.y
                    X = X.x

        if deformation_mode is not None:
            if isinstance(deformation_mode, str):
                deformation_mode = DeformationMode(deformation_mode)
            self._deformation_mode = deformation_mode
            self._poisson_ratio = poisson_ratio

            # Convert E* -> G* if tensile deformation mode
            if deformation_mode.is_tensile() and y is not None:
                from rheojax.utils.modulus_conversion import convert_modulus

                y = convert_modulus(
                    y, deformation_mode, DeformationMode.SHEAR, poisson_ratio
                )
                logger.info(
                    "Converted tensile modulus to shear for fitting",
                    model=self.__class__.__name__,
                    from_mode=str(deformation_mode),
                    poisson_ratio=poisson_ratio,
                )
        else:
            self._deformation_mode = None

        # Store data for potential Bayesian inference
        self.X_data = X
        self.y_data = y

        # Auto-detect optimization strategy
        use_log_residuals, use_multi_start = self._detect_optimization_strategy(
            X, use_log_residuals, use_multi_start, n_starts
        )

        # Pass optimization strategy to _fit via kwargs.
        # These are consumed by _fit() and should NOT leak to model_function.
        # Models' _fit() implementations should pop these before storing _last_fit_kwargs.
        kwargs["use_log_residuals"] = use_log_residuals
        kwargs["use_multi_start"] = use_multi_start
        kwargs["n_starts"] = n_starts
        kwargs["perturb_factor"] = perturb_factor

        # Optional compatibility check before fitting
        test_mode = kwargs.get("test_mode", None)
        if check_compatibility:
            compatibility = self._check_compatibility(X, y, test_mode)
            if compatibility and not compatibility.get("compatible", True):
                try:
                    from rheojax.utils.compatibility import format_compatibility_message

                    message = format_compatibility_message(compatibility)
                    logger.warning(
                        "Model compatibility check failed",
                        model=self.__class__.__name__,
                        message=message,
                    )
                except Exception as exc:
                    logger.debug(
                        "Failed to format compatibility message",
                        error=str(exc),
                    )

        # Call subclass implementation (which uses NLSQ via optimization module)
        try:
            self._fit(X, y, method=method, **kwargs)
            self.fitted_ = True

            # Log fit completion with key metrics
            # Only compute R² when DEBUG logging is active — for ODE models
            # score() triggers a full predict() call costing 100ms+.
            r2 = None
            if logger.isEnabledFor(logging.DEBUG):
                try:
                    r2 = self.score(X, y)
                except Exception as exc:
                    logger.debug(
                        "R² computation failed after fit",
                        model=self.__class__.__name__,
                        error=str(exc),
                    )

            logger.info(
                "Fit completed",
                model=self.__class__.__name__,
                fitted=self.fitted_,
                R2=r2,
                data_shape=data_shape,
            )
            logger.debug(
                "Exiting fit",
                model=self.__class__.__name__,
                parameters=self.get_params(),
            )

        except RuntimeError as e:
            logger.error(
                "Fit failed with RuntimeError",
                model=self.__class__.__name__,
                error=str(e),
                exc_info=True,
            )
            enhanced = self._enhance_error_with_compatibility(e, X, y, test_mode)
            if enhanced is not e:
                raise enhanced from e
            raise
        except Exception as e:
            logger.error(
                "Fit failed with unexpected error",
                model=self.__class__.__name__,
                error=str(e),
                exc_info=True,
            )
            raise

        return self

    def precompile(
        self,
        test_mode: str = "relaxation",
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
    ) -> float:
        """Precompile NLSQ residual functions to eliminate JIT cold-start.

        Triggers JIT compilation by running a minimal fit (``max_iter=1``)
        with dummy data. The model parameters are reset afterwards so
        the model is left in its original state.

        This is useful for interactive sessions or benchmarks where the
        ~870ms first-fit JIT overhead should be excluded.

        Args:
            test_mode: Test mode to precompile for (default: 'relaxation').
            X: Optional input data for shape inference. If None, uses a
                10-point logspace array.
            y: Optional output data. If None, generates ones matching X.

        Returns:
            Compilation time in seconds.

        Example:
            >>> model = Maxwell()
            >>> t = model.precompile(test_mode='relaxation')
            >>> print(f"Compiled in {t:.2f}s")
            >>> model.fit(X, y)  # No JIT overhead
        """
        import time

        logger.info("Starting NLSQ precompilation", model=self.__class__.__name__)

        # Save current state
        saved_params = {
            name: self.parameters.get_value(name) for name in self.parameters
        }
        saved_fitted = self.fitted_

        # Generate dummy data if not provided
        if X is None:
            X = np.logspace(-2, 2, 10, dtype=np.float64)
        X_arr = np.asarray(X, dtype=np.float64)
        if y is None:
            y = np.ones_like(X_arr, dtype=np.float64)

        start_time = time.perf_counter()

        try:
            self._fit(X_arr, y, test_mode=test_mode, max_iter=1)
        except Exception as e:
            logger.warning(
                "NLSQ precompilation fit failed — JIT may still have compiled",
                error=str(e),
            )

        compile_time = time.perf_counter() - start_time

        # Restore original state
        for name, value in saved_params.items():
            if value is not None:
                self.parameters.set_value(name, value)
        self.fitted_ = saved_fitted

        logger.info(
            "NLSQ precompilation completed",
            compile_time_seconds=compile_time,
            model=self.__class__.__name__,
        )

        return compile_time

    def fit_bayesian(  # extends BayesianMixin signature with DMTA params
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        num_warmup: int = 1000,
        num_samples: int = 2000,
        num_chains: int = 4,
        initial_values: dict[str, float] | None = None,
        test_mode: str | None = None,
        seed: int | None = None,
        deformation_mode: str | DeformationMode | None = None,
        poisson_ratio: float = 0.5,
        **nuts_kwargs,
    ) -> BayesianResult:
        """Perform Bayesian inference using NumPyro NUTS sampler.

        This method delegates to BayesianMixin.fit_bayesian() to run NUTS sampling
        for Bayesian parameter estimation. If initial_values is not provided and
        the model has been previously fitted with fit(), the NLSQ point estimates
        are automatically used for warm-starting.

        Multi-chain sampling is enabled by default (num_chains=4) to provide
        reliable convergence diagnostics (R-hat, ESS) and parallel execution
        on multi-GPU systems.

        Args:
            X: Independent variable data (input features) or RheoData object
            y: Dependent variable data (observations to fit). If X is RheoData,
                y is ignored and extracted from X.
            num_warmup: Number of warmup/burn-in iterations (default: 1000)
            num_samples: Number of posterior samples per chain (default: 2000)
            num_chains: Number of MCMC chains (default: 4). Multiple chains
                enable proper R-hat computation and parallel execution.
                Chain method is auto-selected: 'parallel' on multi-GPU,
                'vectorized' on single GPU/CPU.
            initial_values: Optional dict of initial parameter values for
                warm-start. If None and model is fitted, uses NLSQ estimates.
            test_mode: Explicit test mode (e.g., 'relaxation', 'creep', 'oscillation').
                If None, inferred from RheoData.metadata['test_mode'] or defaults
                to 'relaxation'. Overrides RheoData metadata if provided.
            seed: Random seed for reproducibility. If None, uses seed=0 for
                deterministic results. Set to different values for independent runs.
            **nuts_kwargs: Additional arguments passed to NUTS sampler
                (e.g., target_accept_prob, chain_method)

        Returns:
            BayesianResult containing posterior samples, summary statistics,
            and convergence diagnostics (R-hat, ESS, divergences)

        Example:
            >>> model = Maxwell()
            >>> # Warm-start from NLSQ with explicit mode
            >>> model.fit(t, G_data, test_mode='relaxation')  # NLSQ optimization
            >>> result = model.fit_bayesian(t, G_data, test_mode='relaxation')
            >>>
            >>> # RheoData with embedded mode (recommended)
            >>> rheo_data = RheoData(x=omega, y=G_star, metadata={'test_mode': 'oscillation'})
            >>> result = model.fit_bayesian(rheo_data)
            >>>
            >>> # Or provide explicit initial values
            >>> result = model.fit_bayesian(
            ...     t, G_data,
            ...     initial_values={'G0': 1e5, 'eta': 1e3},
            ...     test_mode='creep'
            ... )
        """
        # Get data shape for logging
        data_shape = getattr(X, "shape", None) or (len(X),)
        logger.debug(
            "Entering fit_bayesian",
            model=self.__class__.__name__,
            data_shape=data_shape,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            test_mode=test_mode,
        )

        # Handle deformation mode for Bayesian: convert E* -> G* before NUTS
        if deformation_mode is None:
            from rheojax.core.data import RheoData

            if isinstance(X, RheoData):
                deformation_mode = X.metadata.get("deformation_mode", None)
                if deformation_mode is not None:
                    if y is None:
                        y = X.y
                    # Preserve test_mode from RheoData before stripping to raw array
                    if test_mode is None:
                        test_mode = X.metadata.get("test_mode", None)
                    X = X.x

        if deformation_mode is not None:
            if isinstance(deformation_mode, str):
                deformation_mode = DeformationMode(deformation_mode)
            self._deformation_mode = deformation_mode
            self._poisson_ratio = poisson_ratio

            if deformation_mode.is_tensile() and y is not None:
                from rheojax.utils.modulus_conversion import convert_modulus

                y = convert_modulus(
                    y, deformation_mode, DeformationMode.SHEAR, poisson_ratio
                )
                logger.info(
                    "Converted tensile modulus to shear for Bayesian inference",
                    model=self.__class__.__name__,
                    from_mode=str(deformation_mode),
                    poisson_ratio=poisson_ratio,
                )

        # Store data for model_function access
        self.X_data = X
        self.y_data = y

        # Auto warm-start from NLSQ if available and no explicit initial values
        if initial_values is None and self.fitted_:
            # Extract current parameter values as initial values
            initial_values = {
                name: self.parameters.get_value(name) for name in self.parameters
            }
            logger.debug(
                "Using NLSQ warm-start for Bayesian inference",
                model=self.__class__.__name__,
                initial_values=initial_values,
            )

        # Call BayesianMixin implementation with multi-chain parallelization
        try:
            result = super().fit_bayesian(
                X,
                y,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                initial_values=initial_values,
                test_mode=test_mode,
                seed=seed,
                **nuts_kwargs,
            )

            # Store result for later access
            self._bayesian_result = result

            # Log completion with diagnostics
            r_hat = result.diagnostics.get("r_hat") if result.diagnostics else None
            ess = result.diagnostics.get("ess") if result.diagnostics else None
            logger.info(
                "Bayesian fit completed",
                model=self.__class__.__name__,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                r_hat=r_hat,
                ess=ess,
            )
            logger.debug(
                "Exiting fit_bayesian",
                model=self.__class__.__name__,
                diagnostics=result.diagnostics,
            )

            return result

        except Exception as e:
            logger.error(
                "Bayesian fit failed",
                model=self.__class__.__name__,
                error=str(e),
                exc_info=True,
            )
            raise

    def predict(
        self,
        X: ArrayLike,
        test_mode: str | None = None,
        deformation_mode: str | DeformationMode | None = None,
        poisson_ratio: float | None = None,
        **kwargs,
    ) -> ArrayLike:
        """Make predictions.

        Args:
            X: Input features
            test_mode: Optional test mode ('oscillation', 'relaxation', 'creep', 'flow').
                      If provided, sets model's test_mode before prediction.
                      Useful for data generation without fitting.
            deformation_mode: Optional deformation mode for output conversion.
                If None, uses the mode stored from fit(). If tensile, converts
                G* predictions to E* space.
            poisson_ratio: Poisson's ratio for conversion. If None, uses value
                stored from fit() (default 0.5).
            **kwargs: Additional arguments passed to the internal _predict method.

        Returns:
            Model predictions (in E* space if deformation_mode is tensile)
        """
        x_shape = getattr(X, "shape", None) or (len(X),)
        logger.debug(
            "Predict called",
            model=self.__class__.__name__,
            x_shape=x_shape,
            test_mode=test_mode,
            kwargs=kwargs,
        )

        if not self.fitted_ and len(self.parameters) > 0:
            # Check if we have parameters set manually
            if not any(p.value is None for p in self.parameters._parameters.values()):
                # Parameters are set, consider it fitted
                logger.debug(
                    "Auto-marking model as fitted (parameters set manually)",
                    model=self.__class__.__name__,
                )
                self.fitted_ = True

        # Set test_mode if provided (for data generation without fitting)
        _old_test_mode = getattr(self, "_test_mode", None)
        if test_mode is not None:
            if hasattr(self, "_test_mode"):
                self._test_mode = test_mode
            # Pass test_mode via kwargs for models that read it from kwargs
            kwargs["test_mode"] = test_mode

        try:
            try:
                result = self._predict(X, **kwargs)
            except TypeError as e:
                if "unexpected keyword argument" in str(e):
                    # Progressive kwargs stripping for models whose _predict()
                    # doesn't accept all kwargs. First strip the offending kwarg,
                    # then fall back to bare _predict(X) as last resort.
                    offending = str(e)
                    stripped = dict(kwargs)
                    for key in list(stripped):
                        if key in offending:
                            del stripped[key]
                    try:
                        result = self._predict(X, **stripped)
                    except TypeError:
                        # Final fallback: bare call (13+ models have _predict(self, X) only)
                        result = self._predict(X)
                else:
                    raise

            # Convert G* -> E* if tensile deformation mode
            dm = deformation_mode
            if dm is None:
                dm = self._deformation_mode
            if dm is not None:
                if isinstance(dm, str):
                    dm = DeformationMode(dm)
                if dm.is_tensile():
                    from rheojax.utils.modulus_conversion import convert_modulus

                    nu = (
                        poisson_ratio
                        if poisson_ratio is not None
                        else self._poisson_ratio
                    )
                    result = convert_modulus(result, DeformationMode.SHEAR, dm, nu)

            logger.debug(
                "Predict completed",
                model=self.__class__.__name__,
                output_shape=getattr(result, "shape", None),
            )
            return result
        except Exception as e:
            logger.error(
                "Predict failed",
                model=self.__class__.__name__,
                error=str(e),
                exc_info=True,
            )
            raise
        finally:
            # Restore original test_mode to avoid side effect
            if test_mode is not None and hasattr(self, "_test_mode"):
                self._test_mode = _old_test_mode

    def fit_predict(self, X: ArrayLike, y: ArrayLike, **kwargs) -> ArrayLike:
        """Fit model and return predictions.

        Args:
            X: Input features
            y: Target values
            **kwargs: Additional fitting options

        Returns:
            Model predictions on training data
        """
        logger.debug(
            "fit_predict called",
            model=self.__class__.__name__,
            data_shape=getattr(X, "shape", None) or (len(X),),
        )
        self.fit(X, y, **kwargs)
        return self.predict(X)

    def get_nlsq_result(self):
        """Get stored NLSQ optimization result.

        Returns:
            OptimizationResult from NLSQ fit, or None if not fitted

        Example:
            >>> model.fit(t, G_data)
            >>> result = model.get_nlsq_result()
            >>> if result:
            ...     print(f"Converged: {result.success}")
        """
        return self._nlsq_result

    @property
    def pcov_(self):
        """Parameter covariance matrix from NLSQ fit.

        Returns:
            ndarray of shape (n_params, n_params), or None if not fitted
        """
        return self._nlsq_result.pcov if self._nlsq_result else None

    @property
    def popt_(self):
        """Optimal parameter values from NLSQ fit.

        Returns:
            ndarray of shape (n_params,), or None if not fitted
        """
        return self._nlsq_result.x if self._nlsq_result else None

    def get_parameter_uncertainties(self):
        """Get standard errors for fitted parameters from NLSQ covariance.

        Returns:
            dict of {param_name: std_error}, or None if covariance unavailable
        """
        if self._nlsq_result is None or self._nlsq_result.pcov is None:
            return None
        std_errors = self._nlsq_result.get_parameter_uncertainties()
        if std_errors is None:
            return None
        param_names = list(self.parameters.keys())
        return dict(zip(param_names, std_errors, strict=True))

    def get_bayesian_result(self) -> BayesianResult | None:
        """Get stored Bayesian inference result.

        Returns:
            BayesianResult from fit_bayesian(), or None if not run

        Example:
            >>> model.fit_bayesian(t, G_data)
            >>> result = model.get_bayesian_result()
            >>> print(result.diagnostics['r_hat'])
        """
        return self._bayesian_result

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get model parameters.

        Args:
            deep: If True, return parameters of sub-objects

        Returns:
            Dictionary of parameter names and values
        """
        if hasattr(self, "parameters") and self.parameters:
            return {
                name: self.parameters[name].value for name in self.parameters.keys()
            }
        return {}

    def set_params(self, **params) -> BaseModel:
        """Set model parameters.

        Args:
            **params: Parameter names and values

        Returns:
            self for method chaining
        """
        logger.debug(
            "set_params called",
            model=self.__class__.__name__,
            params=params,
        )
        if hasattr(self, "parameters"):
            for name, value in params.items():
                if name in self.parameters:
                    self.parameters.set_value(name, value)
        return self

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Compute model score (R² by default).

        Args:
            X: Input features
            y: True target values

        Returns:
            Model score (R² coefficient)
        """
        predictions = self.predict(X)

        # Convert to numpy for scoring
        if isinstance(predictions, jnp.ndarray):
            predictions = np.array(predictions)
        if isinstance(y, jnp.ndarray):
            y = np.array(y)

        # Compute R² score
        # For complex data (e.g., oscillatory shear), use magnitude of residuals
        if np.iscomplexobj(y) or np.iscomplexobj(predictions):
            ss_res = np.sum(np.abs(y - predictions) ** 2)
            ss_tot = np.sum(np.abs(y - np.mean(y)) ** 2)
        else:
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)

        # Handle edge cases
        if ss_tot == 0:
            # All y values are constant — R² is undefined
            logger.warning("R² undefined for constant data (ss_tot=0)")
            return np.nan

        # Handle NaN case (e.g. predictions contain NaN/Inf)
        r2 = 1 - (ss_res / ss_tot)
        if np.isnan(r2):
            logger.warning(
                "R² is NaN — predictions may contain NaN/Inf values",
                model=self.__class__.__name__,
            )
            return np.nan

        return float(np.real(r2))

    def to_dict(self) -> dict[str, Any]:
        """Serialize model to dictionary.

        Returns:
            Dictionary representation of model
        """
        return {
            "class": self.__class__.__name__,
            "parameters": self.parameters.to_dict() if self.parameters else {},
            "fitted": self.fitted_,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaseModel:
        """Create model from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Model instance
        """
        model = cls()
        if "parameters" in data:
            model.parameters = ParameterSet.from_dict(data["parameters"])
        model.fitted_ = data.get("fitted", False)
        logger.debug(
            "Model created from dict",
            model=cls.__name__,
            fitted=model.fitted_,
        )
        return model

    def __repr__(self) -> str:
        """String representation of model."""
        params = self.get_params()
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{self.__class__.__name__}({param_str})"


class BaseTransform(ABC):
    """Abstract base class for all data transforms.

    This class defines the standard interface that all transforms must implement,
    supporting JAX arrays and composable transformations.
    """

    def __init__(self):
        """Initialize base transform."""
        logger.debug("Initializing transform", transform=self.__class__.__name__)
        self.fitted_ = False

    @abstractmethod
    def _transform(self, data):
        """Internal transform implementation to be overridden by subclasses.

        Args:
            data: Input data (RheoData or list[RheoData])

        Returns:
            Transformed data (RheoData or tuple[RheoData, dict])
        """
        pass

    def _inverse_transform(self, data):
        """Internal inverse transform implementation.

        Args:
            data: Transformed data (RheoData)

        Returns:
            Original data (RheoData)

        Raises:
            NotImplementedError: If inverse transform not available
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support inverse transform"
        )

    def transform(self, data):
        """Transform the data.

        Args:
            data: Input data (RheoData or list[RheoData])

        Returns:
            Transformed data (RheoData or tuple[RheoData, dict])
        """
        input_shape = getattr(data, "shape", None)
        logger.debug(
            "Entering transform",
            transform=self.__class__.__name__,
            input_shape=input_shape,
        )
        try:
            result = self._transform(data)
            output_shape = getattr(result, "shape", None)
            logger.info(
                "Transform completed",
                transform=self.__class__.__name__,
                input_shape=input_shape,
                output_shape=output_shape,
            )
            logger.debug(
                "Exiting transform",
                transform=self.__class__.__name__,
                output_shape=output_shape,
            )
            return result
        except Exception as e:
            logger.error(
                f"Transform failed: {e}",
                transform=self.__class__.__name__,
                input_shape=input_shape,
                error=str(e),
                exc_info=True,
            )
            raise

    def inverse_transform(self, data):
        """Apply inverse transformation.

        Args:
            data: Transformed data (RheoData)

        Returns:
            Original data (RheoData)
        """
        input_shape = getattr(data, "shape", None)
        logger.debug(
            "Entering inverse_transform",
            transform=self.__class__.__name__,
            input_shape=input_shape,
        )
        try:
            result = self._inverse_transform(data)
            output_shape = getattr(result, "shape", None)
            logger.info(
                "Inverse transform completed",
                transform=self.__class__.__name__,
                input_shape=input_shape,
                output_shape=output_shape,
            )
            logger.debug(
                "Exiting inverse_transform",
                transform=self.__class__.__name__,
                output_shape=output_shape,
            )
            return result
        except Exception as e:
            logger.error(
                f"Inverse transform failed: {e}",
                transform=self.__class__.__name__,
                input_shape=input_shape,
                error=str(e),
                exc_info=True,
            )
            raise

    def fit(self, data) -> BaseTransform:
        """Fit the transform to data (learn parameters if needed).

        Args:
            data: Training data (RheoData or list[RheoData])

        Returns:
            self for method chaining
        """
        input_shape = getattr(data, "shape", None)
        logger.debug(
            "Entering fit",
            transform=self.__class__.__name__,
            input_shape=input_shape,
        )
        # Default implementation does nothing (stateless transform)
        self.fitted_ = True
        logger.info(
            "Transform fit completed",
            transform=self.__class__.__name__,
            input_shape=input_shape,
        )
        logger.debug(
            "Exiting fit",
            transform=self.__class__.__name__,
            fitted=self.fitted_,
        )
        return self

    def fit_transform(self, data):
        """Fit to data and transform it.

        Args:
            data: Input data (RheoData or list[RheoData])

        Returns:
            Transformed data (RheoData or tuple[RheoData, dict])
        """
        input_shape = getattr(data, "shape", None)
        logger.debug(
            "Entering fit_transform",
            transform=self.__class__.__name__,
            input_shape=input_shape,
        )
        self.fit(data)
        result = self.transform(data)
        output_shape = getattr(result, "shape", None)
        logger.info(
            "Fit and transform completed",
            transform=self.__class__.__name__,
            input_shape=input_shape,
            output_shape=output_shape,
        )
        logger.debug(
            "Exiting fit_transform",
            transform=self.__class__.__name__,
            output_shape=output_shape,
        )
        return result

    def __add__(self, other: BaseTransform) -> TransformPipeline:
        """Compose transforms using + operator.

        Args:
            other: Another transform to compose

        Returns:
            Pipeline of transforms
        """
        if isinstance(other, TransformPipeline):
            return TransformPipeline([self] + list(other.transforms))
        elif isinstance(other, BaseTransform):
            return TransformPipeline([self, other])
        else:
            raise TypeError(f"Cannot compose with {type(other)}")

    def __repr__(self) -> str:
        """String representation of transform."""
        return f"{self.__class__.__name__}()"


class TransformPipeline(BaseTransform):
    """Pipeline of multiple transforms applied sequentially."""

    def __init__(self, transforms: list[BaseTransform]):
        """Initialize transform pipeline.

        Args:
            transforms: List of transforms to apply in order
        """
        super().__init__()
        self.transforms = transforms
        logger.debug(
            "Initializing TransformPipeline",
            transform="TransformPipeline",
            n_transforms=len(transforms),
            transform_names=[t.__class__.__name__ for t in transforms],
        )

    def _transform(self, data):
        """Apply all transforms in sequence.

        Args:
            data: Input data (RheoData)

        Returns:
            Transformed data after all transforms
        """
        result = data
        for i, transform in enumerate(self.transforms):
            logger.debug(
                f"Pipeline step {i + 1}/{len(self.transforms)}",
                transform="TransformPipeline",
                step=i + 1,
                total_steps=len(self.transforms),
                current_transform=transform.__class__.__name__,
            )
            step_result = transform.transform(result)
            # Handle tuple returns (data, extras) from mid-pipeline steps
            result = step_result[0] if isinstance(step_result, tuple) else step_result
        return result

    def _inverse_transform(self, data):
        """Apply inverse transforms in reverse order.

        Args:
            data: Transformed data (RheoData)

        Returns:
            Original data (RheoData)
        """
        result = data
        for i, transform in enumerate(reversed(self.transforms)):
            logger.debug(
                f"Pipeline inverse step {i + 1}/{len(self.transforms)}",
                transform="TransformPipeline",
                step=i + 1,
                total_steps=len(self.transforms),
                current_transform=transform.__class__.__name__,
            )
            result = transform.inverse_transform(result)
        return result

    def fit(self, data: ArrayLike) -> TransformPipeline:
        """Fit all transforms in the pipeline.

        Args:
            data: Training data

        Returns:
            self for method chaining
        """
        input_shape = getattr(data, "shape", None)
        logger.debug(
            "Entering pipeline fit",
            transform="TransformPipeline",
            input_shape=input_shape,
            n_transforms=len(self.transforms),
        )
        try:
            current_data = data
            for i, transform in enumerate(self.transforms):
                logger.debug(
                    f"Fitting pipeline step {i + 1}/{len(self.transforms)}",
                    transform="TransformPipeline",
                    step=i + 1,
                    total_steps=len(self.transforms),
                    current_transform=transform.__class__.__name__,
                )
                current_data = transform.fit_transform(current_data)
            self.fitted_ = True
            logger.info(
                "Pipeline fit completed",
                transform="TransformPipeline",
                input_shape=input_shape,
                n_transforms=len(self.transforms),
            )
            logger.debug(
                "Exiting pipeline fit",
                transform="TransformPipeline",
                fitted=self.fitted_,
            )
            return self
        except Exception as e:
            logger.error(
                f"Pipeline fit failed: {e}",
                transform="TransformPipeline",
                input_shape=input_shape,
                n_transforms=len(self.transforms),
                error=str(e),
                exc_info=True,
            )
            raise

    def __repr__(self) -> str:
        """String representation of pipeline."""
        transform_names = " -> ".join(t.__class__.__name__ for t in self.transforms)
        return f"TransformPipeline([{transform_names}])"


__all__ = [
    "BaseModel",
    "BaseTransform",
    "TransformPipeline",
    "Parameter",
    "ParameterSet",
]
