import re

from rheojax.core.base import ArrayLike, BaseModel
from rheojax.core.jax_config import safe_import_jax
from rheojax.utils.optimization import nlsq_optimize

jax, jnp = safe_import_jax()

# Positive scale/stress/viscosity parameters that carry very wide bounds
# (e.g. (0, 1e9), (1e-6, 1e12)). Base names shared by MIKH and the per-mode /
# weighted-sum ML-IKH parameter families (e.g. "delta_sigma_y_2" -> base
# "delta_sigma_y"). Value = LogNormal scale (in log-units, i.e. ~factor e**scale
# per std). Anything not listed keeps the default Uniform-over-bounds prior.
_IKH_LOGNORMAL_PRIOR_SCALES = {
    "G": 1.0,
    "eta": 1.5,
    "C": 1.0,
    "sigma_y0": 1.0,
    "delta_sigma_y": 1.0,
    "eta_inf": 1.5,
    "tau_thix": 1.5,
    "k3": 1.0,
}


class IKHBase(BaseModel):
    """Base class for Isotropic-Kinematic Hardening models.

    These models describe elasto-viscoplastic behavior with:
    1. Kinematic Hardening (Armstrong-Frederick type) - backstress evolution
    2. Isotropic Hardening/Softening (Thixotropy) - yield surface evolution
    """

    def __init__(self):
        super().__init__()

    def bayesian_prior_factory(self, name, lower, upper):
        """Weakly-informative priors for wide-bound positive scale parameters.

        The default Bayesian prior is Uniform over each parameter's bounds. For
        the stress/viscosity/timescale parameters those bounds span up to 9-18
        decades (e.g. delta_sigma_y in (0, 1e9)), which is effectively improper.
        The IKH flow curve has a collinear/flat likelihood direction (tau_thix
        can grow so lambda_ss -> 0, leaving delta_sigma_y unconstrained), so a
        near-flat Uniform lets NUTS drift to ~1e8 and produces divergences.

        Replacing that with a LogNormal centered on the NLSQ warm-start value
        (the same warm start already used to initialize NUTS) keeps these
        parameters within a couple decades of the point estimate while staying
        broad. Mirrors the SPP yield-stress model's prior factory. Returns None
        (default Uniform) for exponent/unit parameters and the noise sigma.
        """
        base = re.sub(r"_\d+$", "", name)
        scale = _IKH_LOGNORMAL_PRIOR_SCALES.get(base)
        if scale is None:
            return None

        import math

        import numpyro.distributions as dist

        # Center on the current (NLSQ-fitted, or __init__ default) value.
        # NOTE: use Python math (not jnp) — this runs inside NUTS's jitted trace,
        # where any jnp op on a concrete float still yields a tracer and would
        # break float()/LogNormal's concrete-loc requirement.
        p = self.parameters.get(name)
        center = float(p.value) if (p is not None and p.value is not None) else None
        if center is None or not (center > 0):
            # Fall back to the geometric midpoint of positive bounds.
            lo = float(lower) if lower is not None else None
            hi = float(upper) if upper is not None else None
            if lo is not None and hi is not None and lo > 0 and hi > 0:
                center = math.sqrt(lo * hi)
            else:
                center = 1.0
        return dist.LogNormal(loc=math.log(center), scale=scale)

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

        # For startup/laos: auto-compute strain from gamma_dot
        # strain = gamma_dot * time for constant shear rate startup
        if "gamma_dot" in kwargs:
            gamma_dot = kwargs["gamma_dot"]
            strain = X_arr * gamma_dot
            return X_arr, strain

        # For LAOS: auto-compute strain from gamma_0 and omega
        # strain = gamma_0 * sin(omega * t)
        if "gamma_0" in kwargs and "omega" in kwargs:
            gamma_0 = kwargs["gamma_0"]
            omega = kwargs["omega"]
            strain = gamma_0 * jnp.sin(omega * X_arr)
            return X_arr, strain

        # If X is time (N,) and no strain provided, this is likely an error for these models
        # unless it's a specific protocol (e.g. constant shear rate implicit).
        # For now, require explicit strain.
        raise ValueError(
            "IKH models require both time and strain history. "
            "Pass RheoData, or X of shape (2, N), or X=time with strain=gamma kwarg, "
            "or gamma_dot for startup, or gamma_0+omega for LAOS."
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
            p_dict = dict(zip(p_names, param_values, strict=True))

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
