from rheojax.core.base import ArrayLike, BaseModel
from rheojax.core.jax_config import safe_import_jax
from rheojax.utils.optimization import nlsq_optimize

jax, jnp = safe_import_jax()


class IKHBase(BaseModel):
    """Base class for Isotropic-Kinematic Hardening models.

    These models describe elasto-viscoplastic behavior with:
    1. Kinematic Hardening (Armstrong-Frederick type) - backstress evolution
    2. Isotropic Hardening/Softening (Thixotropy) - yield surface evolution
    """

    def __init__(self):
        super().__init__()

    def _extract_time_strain(self, X, **kwargs):
        """Helper to extract time and strain from inputs.

        Args:
            X: RheoData, or [time, strain] array, or time array.
            **kwargs: May contain 'strain' if X is only time.

        Returns:
            (times, strains) as jnp arrays
        """
        if hasattr(X, "time") and hasattr(X, "strain"):
            # RheoData
            return jnp.asarray(X.time), jnp.asarray(X.strain)

        X_arr = jnp.asarray(X)

        # If X has shape (2, N), assume [time, strain]
        if X_arr.ndim == 2 and X_arr.shape[0] == 2:
            return X_arr[0], X_arr[1]

        # If X is time (N,), look for strain in kwargs
        if "strain" in kwargs:
            return X_arr, jnp.asarray(kwargs["strain"])

        # If X is time (N,) and no strain provided, this is likely an error for these models
        # unless it's a specific protocol (e.g. constant shear rate implicit).
        # For now, require explicit strain.
        raise ValueError(
            "IKH models require both time and strain history. "
            "Pass RheoData, or X of shape (2, N), or X=time with strain=gamma kwarg."
        )

    def _fit(self, X: ArrayLike, y: ArrayLike, **kwargs) -> "IKHBase":
        """Fit model parameters to data.

        Args:
            X: Time/Strain input.
            y: Stress output (target).
            **kwargs: Optimization options.
        """
        # Prepare data for objective function
        # We need a wrapper that adapts the generic (X, params) signature to our needs
        # create_least_squares_objective expects model_fn(X, params) -> y_pred

        # However, our _predict requires extraction logic which might not be clean
        # if passed just X array to the objective.
        # Let's define a custom objective closure here.

        # 1. Extract data once
        times, strains = self._extract_time_strain(X, **kwargs)
        # Combine for internal passing if needed, but easier to close over them

        y_target = jnp.asarray(y)

        # 2. Define objective for NLSQ
        # nlsq_optimize expects: objective(param_values) -> residuals_vector
        def objective(param_values):
            # Update temporary parameter set or reconstruct dict
            # Use self.model_function for stateless prediction

            # Note: model_function expects 'params' which matches the structure
            # returned by self.parameters.get_values() IF we are consistent.
            # Base implementation:

            # If model_function expects the flat array from parameters.get_values(), we are good.
            # But specific models might need to restructure (like MLIKH).
            # The BayesianMixin assumes model_function takes what parameters.get_values() returns (mostly).
            # Actually NumPyro samples individual parameters.

            # Let's assume model_function handles the "flat array to dict/structure" conversion
            # OR we rely on specific model implementation.

            # For MIKH: model_function takes dict? No, MIKH.model_function took **params.
            # We need to standardize `model_function` to take an array for NLSQ
            # OR wrap it here.

            # Let's parse param_values back to dict for the kernel
            p_names = list(self.parameters.keys())
            p_dict = dict(zip(p_names, param_values, strict=False))

            # Call kernel
            # We can't easily call model_function if it has different signature requirements.
            # Let's simply call the kernel directly here or via a protected method.

            # It's better to delegate to `self.model_function` but ensure it accepts the array/dict correctly.
            # Let's update `model_function` in subclasses to accept a DICT of values (from NumPyro)
            # AND handle array input here.

            # Actually, standardizing on: model_function(X, params_array_or_dict)

            if hasattr(self, "_predict_from_params"):
                y_pred = self._predict_from_params(times, strains, p_dict)
            else:
                # Fallback: manual map for now if subclasses don't implement _predict_from_params
                # MIKH and MLIKH should implement _predict_from_params
                raise NotImplementedError(
                    "Subclasses must implement _predict_from_params"
                )

            return y_pred - y_target

        # 3. Optimize
        nlsq_optimize(objective, self.parameters, **kwargs)

        return self
