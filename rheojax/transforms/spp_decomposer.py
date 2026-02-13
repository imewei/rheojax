"""SPP Decomposer transform for LAOS stress decomposition.

This module implements the SPP (Sequence of Physical Processes) decomposition
transform for analyzing Large Amplitude Oscillatory Shear (LAOS) data. The
transform decomposes stress signals into elastic and viscous contributions
and extracts yield stress parameters.

The SPP framework provides cycle-by-cycle analysis of nonlinear viscoelastic
responses, enabling extraction of physically meaningful material parameters
from LAOS experiments.

Key Outputs
-----------
- G_cage: Time-resolved apparent cage modulus
- sigma_sy: Static yield stress (at strain reversal)
- sigma_dy: Dynamic yield stress (at rate reversal)
- K, n: Power-law flow parameters
- Lissajous metrics: S-factor, T-factor
- Harmonic amplitudes and phases

References
----------
- S.A. Rogers et al., J. Rheol. 56(1), 2012
- S.A. Rogers, Rheol. Acta 56, 2017
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from jax import Array

from rheojax.core.base import BaseTransform
from rheojax.core.inventory import TransformType
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.registry import TransformRegistry
from rheojax.logging import get_logger

# Safe JAX import (enforces float64)
_, jnp = safe_import_jax()

# Module logger
logger = get_logger(__name__)

if TYPE_CHECKING:

    from rheojax.core.data import RheoData


@TransformRegistry.register("spp_decomposer", type=TransformType.DECOMPOSITION)
class SPPDecomposer(BaseTransform):
    """SPP decomposition transform for LAOS stress analysis.

    Applies the Sequence of Physical Processes (SPP) framework to decompose
    LAOS stress signals and extract nonlinear viscoelastic parameters.

    The transform requires oscillatory shear data with known frequency and
    strain amplitude. It computes:

    1. Elastic/viscous stress decomposition
    2. Yield stress extraction (static and dynamic)
    3. Power-law flow parameters
    4. Lissajous-Bowditch metrics
    5. Harmonic decomposition

    Parameters
    ----------
    omega : float
        Angular frequency ω (rad/s) of the oscillation
    gamma_0 : float
        Strain amplitude γ_0 (dimensionless)
    n_harmonics : int, optional
        Number of odd harmonics to extract for stress (default: 39 per MATLAB SPPplus)
    yield_tolerance : float, optional
        Fractional tolerance for yield point detection (default: 0.02)
    start_cycle : int, optional
        First cycle to analyze (0-indexed). Use to skip startup transients.
        Default: 0 (start from beginning).
    end_cycle : int or None, optional
        Last cycle to analyze (exclusive). None means use all available cycles.
        Default: None.
    use_numerical_method : bool, optional
        If True, use MATLAB-compatible numerical differentiation for raw data.
        If False (default), use Fourier-based decomposition.
    step_size : int, optional
        Step size k for numerical differentiation (only used if use_numerical_method=True).
        Default: 8 to mirror SPPplus v2.1.
    num_mode : int, optional
        Numerical differentiation mode (1 = edge-aware, 2 = periodic/looped),
        matching SPPplus `num_mode`. Only used when use_numerical_method=True.

    Attributes
    ----------
    omega : float
        Angular frequency
    gamma_0 : float
        Strain amplitude
    gamma_dot_0 : float
        Strain rate amplitude (ω * γ_0)
    n_harmonics : int
        Number of harmonics for decomposition
    start_cycle : int
        First cycle to analyze
    end_cycle : int or None
        Last cycle to analyze
    use_numerical_method : bool
        Whether using numerical differentiation
    results_ : dict
        Dictionary of computed SPP metrics (after transform)

    Examples
    --------
    Basic usage with RheoData:

    >>> from rheojax.core.data import RheoData
    >>> from rheojax.transforms.spp_decomposer import SPPDecomposer
    >>>
    >>> # LAOS stress-strain data
    >>> omega = 1.0  # rad/s
    >>> gamma_0 = 1.0  # strain amplitude
    >>> t = jnp.linspace(0, 2*jnp.pi, 1000)
    >>> strain = gamma_0 * jnp.sin(omega * t)
    >>> stress = 100.0 * strain + 20.0 * jnp.sin(3 * omega * t)  # With 3rd harmonic
    >>>
    >>> data = RheoData(
    ...     x=t,
    ...     y=stress,
    ...     domain='time',
    ...     metadata={
    ...         'test_mode': 'oscillation',
    ...         'omega': omega,
    ...         'gamma_0': gamma_0,
    ...         'strain': strain,
    ...     }
    ... )
    >>>
    >>> # Apply SPP decomposition
    >>> decomposer = SPPDecomposer(omega=omega, gamma_0=gamma_0)
    >>> result = decomposer.transform(data)
    >>>
    >>> # Access metrics
    >>> print(f"Static yield stress: {decomposer.results_['sigma_sy']:.2f} Pa")
    >>> print(f"Dynamic yield stress: {decomposer.results_['sigma_dy']:.2f} Pa")

    Notes
    -----
    - Input data must be time-domain stress signal
    - Strain data must be available in metadata['strain'] or computed from ω, γ_0
    - Output includes both decomposed waveforms and extracted parameters
    """

    def __init__(
        self,
        omega: float,
        gamma_0: float,
        n_harmonics: int = 39,
        yield_tolerance: float = 0.02,
        start_cycle: int = 0,
        end_cycle: int | None = None,
        use_numerical_method: bool = False,
        step_size: int = 8,
        num_mode: int = 2,
        wrap_strain_rate: bool = True,
    ):
        """Initialize SPP decomposer transform.

        Parameters
        ----------
        omega : float
            Angular frequency (rad/s)
        gamma_0 : float
            Strain amplitude (dimensionless)
        n_harmonics : int, optional
            Number of odd harmonics to extract (default: 39)
        yield_tolerance : float, optional
            Tolerance for yield point detection (default: 0.02)
        start_cycle : int, optional
            First cycle to analyze, 0-indexed (default: 0)
        end_cycle : int or None, optional
            Last cycle to analyze, exclusive (default: None, use all)
        use_numerical_method : bool, optional
            Use MATLAB-compatible numerical differentiation (default: False)
        step_size : int, optional
            Step size k for numerical differentiation (default: 8)
        num_mode : int, optional
            Numerical differentiation mode (1=edge-aware, 2=periodic). Default: 2.
        wrap_strain_rate : bool, optional
            If True, infer strain rate with periodic wrapping when missing (default: True)
        """
        super().__init__()
        self.omega = float(omega)
        self.gamma_0 = float(gamma_0)
        self.gamma_dot_0 = self.omega * self.gamma_0  # Rate amplitude
        self.n_harmonics = n_harmonics
        self.yield_tolerance = yield_tolerance
        self.start_cycle = start_cycle
        self.end_cycle = end_cycle
        self.use_numerical_method = use_numerical_method
        self.step_size = step_size
        self.num_mode = num_mode
        self.wrap_strain_rate = wrap_strain_rate
        self.results_: dict = {}

    def _get_cycle_mask(
        self,
        t: Array,
    ) -> tuple[Array, int, int]:
        """Compute mask for selected cycles from time series data.

        Parameters
        ----------
        t : jnp.ndarray
            Time array

        Returns
        -------
        mask : jnp.ndarray
            Boolean mask for selected cycles
        actual_start : int
            Actual start cycle index used
        actual_end : int
            Actual end cycle index used
        """
        # Compute period and number of cycles
        T_period = 2 * jnp.pi / self.omega
        total_time = float(t[-1] - t[0])
        n_cycles_total = max(1, int(total_time / T_period))

        # Determine cycle range
        actual_start = max(0, min(self.start_cycle, n_cycles_total - 1))
        if self.end_cycle is None:
            actual_end = n_cycles_total
        else:
            actual_end = min(self.end_cycle, n_cycles_total)

        # If no valid range, use all data
        if actual_start >= actual_end:
            mask = jnp.ones(len(t), dtype=bool)
            return mask, 0, n_cycles_total

        # Calculate time bounds for selected cycles
        t_start = float(t[0]) + actual_start * T_period
        t_end = float(t[0]) + actual_end * T_period

        # Select indices within the cycle range
        mask = (t >= t_start) & (t < t_end)

        return mask, actual_start, actual_end

    def _transform(self, data: RheoData) -> RheoData:  # type: ignore[override]
        """Apply SPP decomposition to LAOS stress data.

        Parameters
        ----------
        data : RheoData
            Time-domain stress data with strain in metadata

        Returns
        -------
        RheoData
            Decomposed stress data with SPP metrics in metadata

        Raises
        ------
        ValueError
            If data is not time-domain or missing required metadata
        """
        from rheojax.core.data import RheoData
        from rheojax.utils.spp_kernels import (
            apparent_cage_modulus,
            build_spp_exports,
            differentiate_rate_from_strain,
            dynamic_yield_stress,
            harmonic_reconstruction,
            lissajous_metrics,
            power_law_fit,
            spp_fourier_analysis,
            spp_numerical_analysis,
            spp_stress_decomposition,
            static_yield_stress,
        )

        logger.info(
            "Starting SPP decomposition",
            omega=self.omega,
            gamma_0=self.gamma_0,
            n_harmonics=self.n_harmonics,
            use_numerical_method=self.use_numerical_method,
        )

        # Validate domain
        if data.domain != "time":
            logger.error(
                "Invalid domain for SPP decomposer",
                expected="time",
                got=data.domain,
            )
            raise ValueError(
                f"SPP decomposer requires time-domain data, got '{data.domain}'"
            )

        # Get time and stress arrays
        t = data.x
        stress = data.y

        # Validate time steps uniformity
        if len(t) > 2:  # type: ignore[arg-type]
            dt_all = np.diff(t)  # type: ignore[arg-type]
            dt_mean = np.mean(dt_all)
            dt_std = np.std(dt_all)
            if dt_mean > 0 and (dt_std / dt_mean > 0.05):  # 5% tolerance
                logger.warning(
                    "Non-uniform time steps detected in SPP data",
                    dt_mean=float(dt_mean),
                    dt_std=float(dt_std),
                    cv=float(dt_std / dt_mean),
                )

        logger.debug(
            "Input data extracted",
            data_points=len(t),  # type: ignore[arg-type]
            domain=data.domain,
        )

        # Convert to JAX arrays
        t_jax = jnp.asarray(t, dtype=jnp.float64)
        stress_jax = jnp.asarray(stress, dtype=jnp.float64)

        # Handle complex stress (take real part)
        if jnp.iscomplexobj(stress_jax):
            logger.debug("Converting complex stress to real part")
            stress_jax = jnp.real(stress_jax)

        # Resolve omega (scalar or per-sample) from metadata if provided
        omega_meta = (
            data.metadata.get("omega", self.omega) if data.metadata else self.omega
        )
        omega_jax = jnp.asarray(omega_meta, dtype=jnp.float64)
        if omega_jax.ndim == 0:
            omega_jax = jnp.full_like(t_jax, omega_jax)
        omega_scalar = float(jnp.mean(omega_jax))

        # Get or compute strain and strain rate
        if "strain" in data.metadata:
            strain_jax = jnp.asarray(data.metadata["strain"], dtype=jnp.float64)
        else:
            # Generate strain from sinusoidal assumption using mean omega
            strain_jax = self.gamma_0 * jnp.sin(omega_scalar * t_jax)

        if "strain_rate" in data.metadata:
            strain_rate_jax = jnp.asarray(
                data.metadata["strain_rate"], dtype=jnp.float64
            )
        else:
            # Compute strain rate via wrapped differentiation (Rogers parity)
            strain_rate_jax = differentiate_rate_from_strain(
                strain_jax,
                float(t_jax[1] - t_jax[0]) if len(t_jax) > 1 else 0.001,
                step_size=self.step_size,
                looped=self.wrap_strain_rate,
            )

        self.gamma_dot_0 = omega_scalar * self.gamma_0

        # =====================================================================
        # Cycle Selection
        # =====================================================================

        # Apply cycle selection if specified
        if self.start_cycle > 0 or self.end_cycle is not None:
            logger.debug(
                "Applying cycle selection",
                start_cycle=self.start_cycle,
                end_cycle=self.end_cycle,
            )
            # Get a single mask and apply to all arrays consistently
            mask, actual_start, actual_end = self._get_cycle_mask(t_jax)

            # Apply mask to all arrays
            t_jax = t_jax[mask]
            stress_jax = stress_jax[mask]
            strain_jax = strain_jax[mask]
            strain_rate_jax = strain_rate_jax[mask]

            logger.debug(
                "Cycle selection applied",
                actual_start=actual_start,
                actual_end=actual_end,
                selected_points=int(jnp.sum(mask)),
            )
        else:
            actual_start, actual_end = 0, None

        # =====================================================================
        # SPP Analysis
        # =====================================================================

        # Compute dt for analyses
        dt = float(t_jax[1] - t_jax[0]) if len(t_jax) > 1 else 0.001

        # Initialize method results (may be populated below)
        core_results = None
        fsf_data_out = None
        ft_out = None
        spp_params = None

        # Number of cycles observed (after masking)
        n_cycles_obs = max(
            1,
            int(
                jnp.round(
                    (float(t_jax[-1]) - float(t_jax[0])) / (2 * jnp.pi / omega_scalar)
                )
            ),
        )

        if self.use_numerical_method:
            logger.debug(
                "Using numerical SPP analysis",
                step_size=self.step_size,
                num_mode=self.num_mode,
            )
            core_results = spp_numerical_analysis(
                strain_jax,
                stress_jax,
                omega_jax,
                dt,
                step_size=self.step_size,
                num_mode=self.num_mode,
            )
            fsf_data_out = core_results["fsf_data_out"]
            spp_params = np.array(
                [
                    float(self.omega),
                    np.nan,
                    np.nan,
                    np.nan,
                    float(self.step_size),
                    float(self.num_mode),
                ]
            )
            ft_out = None
        else:
            logger.debug(
                "Using Fourier SPP analysis",
                n_harmonics=self.n_harmonics,
                n_cycles=n_cycles_obs,
            )
            core_results = spp_fourier_analysis(
                strain_jax,
                stress_jax,
                omega_scalar,
                dt,
                n_harmonics=self.n_harmonics,
                n_cycles=n_cycles_obs,
            )
            fsf_data_out = core_results["fsf_data_out"]
            ft_out = core_results.get("ft_out")
            W = int(round(len(strain_jax) / (2 * n_cycles_obs)))
            spp_params = np.array(
                [
                    omega_scalar,
                    int(self.n_harmonics),
                    int(n_cycles_obs),
                    W,
                    np.nan,
                    np.nan,
                ]
            )

        # 1. Apparent cage modulus
        logger.debug("Computing apparent cage modulus")
        G_cage = apparent_cage_modulus(stress_jax, strain_jax, self.gamma_0)

        # 2. Static yield stress (at |γ| ≈ γ_0)
        logger.debug("Computing static yield stress", tolerance=self.yield_tolerance)
        sigma_sy = static_yield_stress(
            stress_jax, strain_jax, self.gamma_0, tolerance=self.yield_tolerance
        )

        # 3. Dynamic yield stress (at |γ̇| ≈ 0)
        logger.debug("Computing dynamic yield stress", tolerance=self.yield_tolerance)
        sigma_dy = dynamic_yield_stress(
            stress_jax,
            strain_rate_jax,
            self.gamma_dot_0,
            tolerance=self.yield_tolerance,
        )

        # 4. Harmonic reconstruction (for reporting) - stress only
        logger.debug("Performing harmonic reconstruction", n_harmonics=self.n_harmonics)
        amplitudes, phases, stress_reconstructed = harmonic_reconstruction(
            stress_jax, self.omega, n_harmonics=self.n_harmonics, dt=dt
        )

        # 5. Power-law fit
        logger.debug("Fitting power-law model")
        K, n_power, r_squared_power = power_law_fit(stress_jax, strain_rate_jax)

        # 6. Lissajous metrics
        logger.debug("Computing Lissajous metrics")
        lissajous = lissajous_metrics(
            stress_jax,
            strain_jax,
            strain_rate_jax,
            self.gamma_0,
            self.gamma_dot_0,
        )

        # 7. Stress decomposition
        logger.debug("Performing stress decomposition")
        sigma_elastic, sigma_viscous = spp_stress_decomposition(
            stress_jax,
            strain_jax,
            strain_rate_jax,
            self.gamma_0,
            self.gamma_dot_0,
        )

        # =====================================================================
        # Store Results
        # =====================================================================

        self.results_ = {
            # Yield stresses
            "sigma_sy": float(sigma_sy),
            "sigma_dy": float(sigma_dy),
            # Power-law parameters
            "K": float(K),
            "n_power_law": float(n_power),
            "r_squared_power_law": float(r_squared_power),
            # Harmonic analysis
            "harmonic_amplitudes": np.array(amplitudes),
            "harmonic_phases": np.array(phases),
            "fundamental_amplitude": (
                float(amplitudes[0]) if len(amplitudes) > 0 else 0.0
            ),
            "I3_I1_ratio": (
                float(amplitudes[1] / amplitudes[0])
                if len(amplitudes) > 1 and amplitudes[0] > 1e-10
                else 0.0
            ),
            # Lissajous metrics
            "G_L": float(lissajous["G_L"]),
            "G_M": float(lissajous["G_M"]),
            "eta_L": float(lissajous["eta_L"]),
            "eta_M": float(lissajous["eta_M"]),
            "S_factor": float(lissajous["S_factor"]),
            "T_factor": float(lissajous["T_factor"]),
            # Waveforms
            "G_cage": np.array(G_cage),
            "sigma_elastic": np.array(sigma_elastic),
            "sigma_viscous": np.array(sigma_viscous),
            "stress_reconstructed": np.array(stress_reconstructed),
            # Cycle selection info
            "cycles_analyzed": (actual_start, actual_end),
        }
        if core_results is not None:
            core_block = {
                k: np.array(core_results[k])
                for k in [
                    "Gp_t",
                    "Gpp_t",
                    "G_star_t",
                    "tan_delta_t",
                    "delta_t",
                    "disp_stress",
                    "eq_strain_est",
                    "Gp_t_dot",
                    "Gpp_t_dot",
                    "G_speed",
                    "delta_t_dot",
                    "strain_recon",
                    "rate_recon",
                    "stress_recon",
                    "time_new",
                ]
                if k in core_results
            }
            if "Delta" in core_results:
                self.results_["Delta"] = float(core_results["Delta"])
            self.results_["core"] = core_block
            self.results_["spp_params"] = spp_params
            if fsf_data_out is not None:
                self.results_["fsf_data_out"] = np.array(fsf_data_out)
            if ft_out is not None:
                self.results_["ft_out"] = np.array(ft_out)

            # Build MATLAB-compatible tables (15 cols + optional FSF)
            spp_export = build_spp_exports(
                np.array(core_results.get("time_new", t_jax)),
                np.array(core_results.get("strain_recon", strain_jax)),
                np.array(core_results.get("rate_recon", strain_rate_jax)),
                np.array(core_results.get("stress_recon", stress_jax)),
                core_results,
                fsf_data_out,
                spp_params,
            )
            self.results_["spp_data_out"] = spp_export["spp_data_out"]
            if spp_export["fsf_data_out"] is not None:
                self.results_["fsf_data_out"] = spp_export["fsf_data_out"]

            # Mean values for convenience
            self.results_["Gp_t_mean"] = float(jnp.nanmean(core_results["Gp_t"]))
            self.results_["Gpp_t_mean"] = float(jnp.nanmean(core_results["Gpp_t"]))

            # Backward compatibility: expose numerical block when numerical method used
            if self.use_numerical_method:
                self.results_["numerical"] = core_block

        # Create output RheoData with decomposed stress
        new_metadata = data.metadata.copy() if data.metadata else {}
        new_metadata.update(
            {
                "transform": "spp_decomposer",
                "spp_results": self.results_,
                "omega": self.omega,
                "gamma_0": self.gamma_0,
                "n_harmonics": self.n_harmonics,
                "start_cycle": actual_start,
                "end_cycle": actual_end,
                "use_numerical_method": self.use_numerical_method,
                "step_size": self.step_size,
                "num_mode": self.num_mode,
            }
        )

        logger.info(
            "SPP decomposition completed",
            sigma_sy=float(sigma_sy),
            sigma_dy=float(sigma_dy),
            K=float(K),
            n_power_law=float(n_power),
            S_factor=float(lissajous["S_factor"]),
            T_factor=float(lissajous["T_factor"]),
            cycles_analyzed=(actual_start, actual_end),
        )

        # Output: reconstructed stress (or original stress with metrics attached)
        return RheoData(
            x=np.array(t_jax),
            y=np.array(stress_reconstructed),
            x_units=data.x_units or "s",
            y_units=data.y_units or "Pa",
            domain="time",
            metadata=new_metadata,
            validate=False,
        )

    def get_results(self) -> dict:
        """Get computed SPP analysis results.

        Returns
        -------
        dict
            Dictionary containing all SPP metrics:
            - sigma_sy: Static yield stress (Pa)
            - sigma_dy: Dynamic yield stress (Pa)
            - K: Consistency index (Pa·s^n)
            - n_power_law: Power-law exponent
            - harmonic_amplitudes: Array of harmonic amplitudes
            - harmonic_phases: Array of harmonic phases
            - I3_I1_ratio: Third harmonic nonlinearity ratio
            - G_L, G_M: Large and minimum strain moduli (Pa)
            - eta_L, eta_M: Large and minimum rate viscosities (Pa·s)
            - S_factor: Stiffening ratio
            - T_factor: Thickening ratio
            - G_cage: Time-resolved cage modulus (array)
            - sigma_elastic: Elastic stress contribution (array)
            - sigma_viscous: Viscous stress contribution (array)

        Raises
        ------
        RuntimeError
            If transform has not been applied yet

        Examples
        --------
        >>> decomposer = SPPDecomposer(omega=1.0, gamma_0=1.0)
        >>> _ = decomposer.transform(data)
        >>> results = decomposer.get_results()
        >>> print(f"I3/I1 = {results['I3_I1_ratio']:.4f}")
        """
        if not self.results_:
            raise RuntimeError("Transform not yet applied. Call transform() first.")
        return self.results_.copy()

    def get_yield_stresses(self) -> tuple[float, float]:
        """Get static and dynamic yield stresses.

        Returns
        -------
        tuple[float, float]
            (sigma_sy, sigma_dy) in Pa

        Raises
        ------
        RuntimeError
            If transform has not been applied yet
        """
        if not self.results_:
            raise RuntimeError("Transform not yet applied. Call transform() first.")
        return self.results_["sigma_sy"], self.results_["sigma_dy"]

    def get_nonlinearity_metrics(self) -> dict:
        """Get nonlinearity quantification metrics.

        Returns
        -------
        dict
            Dictionary with:
            - I3_I1_ratio: Third harmonic ratio (FT-rheology)
            - S_factor: Strain stiffening ratio
            - T_factor: Shear thickening ratio

        Raises
        ------
        RuntimeError
            If transform has not been applied yet
        """
        if not self.results_:
            raise RuntimeError("Transform not yet applied. Call transform() first.")
        return {
            "I3_I1_ratio": self.results_["I3_I1_ratio"],
            "S_factor": self.results_["S_factor"],
            "T_factor": self.results_["T_factor"],
        }

    def __repr__(self) -> str:
        """String representation of transform."""
        return (
            f"SPPDecomposer(omega={self.omega}, gamma_0={self.gamma_0}, "
            f"n_harmonics={self.n_harmonics})"
        )


def spp_analyze(
    stress: np.ndarray,
    time: np.ndarray,
    omega: float,
    gamma_0: float,
    strain: np.ndarray | None = None,
    n_harmonics: int = 5,
) -> dict:
    """Convenience function for single-shot SPP analysis.

    A standalone function for quick SPP analysis without creating RheoData.
    Useful for scripts and exploratory analysis.

    Parameters
    ----------
    stress : np.ndarray
        Stress signal (Pa)
    time : np.ndarray
        Time array (s)
    omega : float
        Angular frequency (rad/s)
    gamma_0 : float
        Strain amplitude (dimensionless)
    strain : np.ndarray, optional
        Strain signal. If None, computed from sinusoidal assumption.
    n_harmonics : int, optional
        Number of harmonics (default: 5)

    Returns
    -------
    dict
        Complete SPP analysis results including:
        - yield stresses (sigma_sy, sigma_dy)
        - power-law parameters (K, n)
        - harmonic analysis
        - Lissajous metrics
        - decomposed waveforms

    Examples
    --------
    >>> import numpy as np
    >>> from rheojax.transforms.spp_decomposer import spp_analyze
    >>>
    >>> omega = 1.0
    >>> gamma_0 = 1.0
    >>> t = np.linspace(0, 2*np.pi, 1000)
    >>> stress = 100.0 * np.sin(omega * t)
    >>>
    >>> results = spp_analyze(stress, t, omega, gamma_0)
    >>> print(f"Static yield stress: {results['sigma_sy']:.2f} Pa")
    """
    from rheojax.core.data import RheoData

    # Build metadata
    metadata = {
        "test_mode": "oscillation",
        "omega": omega,
        "gamma_0": gamma_0,
    }

    if strain is not None:
        metadata["strain"] = strain

    # Create RheoData
    data = RheoData(
        x=time,
        y=stress,
        domain="time",
        metadata=metadata,
        validate=False,
    )

    # Apply decomposer
    decomposer = SPPDecomposer(omega=omega, gamma_0=gamma_0, n_harmonics=n_harmonics)
    decomposer.transform(data)  # type: ignore[arg-type]

    return decomposer.get_results()


__all__ = ["SPPDecomposer", "spp_analyze"]
