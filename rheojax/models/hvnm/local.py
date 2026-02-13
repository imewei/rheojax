"""HVNMLocal: Local (0D) Hybrid Vitrimer Nanocomposite Model.

Single-point constitutive model for NP-filled vitrimers with four subnetworks:
1. Permanent (P): covalent crosslinks, amplified by Guth-Gold X(phi)
2. Exchangeable (E): associative vitrimer bonds with matrix BER kinetics
3. Dissociative (D): physical reversible bonds, standard Maxwell
4. Interphase (I): NP-bound chains with distinct interfacial BER kinetics

Supports 6 rheological protocols:
- Flow curve (analytical: sigma_E -> 0, sigma_I -> 0 at steady state)
- SAOS (analytical: three Maxwell modes + amplified permanent plateau)
- Startup shear (ODE with dual TST feedback)
- Stress relaxation (ODE: quad-exponential + amplified plateau)
- Creep (ODE: implicit gamma_dot solve, 4-network stress balance)
- LAOS (ODE: nonlinear oscillatory response with Payne effect)

Limiting Cases
--------------
- phi=0: Recovers HVM exactly (primary validation criterion)
- G_E=0, G_D=0, G_I=0: Amplified neo-Hookean
- G_P=0, G_E=0, G_I=0: Maxwell fluid
- k_BER^int -> 0: Frozen interphase (dead layer)
- Full: 4-network HVNM

References
----------
- Vernerey, Long, & Brighenti (2017). JMPS 107, 1-20.
- Li, Zhao, Duan, Zhang, Liu (2024). Langmuir 40, 7550-7560.
- Karim, Vernerey, Sain (2025). Macromolecules 58, 4899-4912.
"""

from __future__ import annotations

import logging
from typing import Literal

import diffrax
import numpy as np

from rheojax.core.inventory import Protocol
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.registry import ModelRegistry
from rheojax.core.test_modes import DeformationMode
from rheojax.models.hvnm._base import HVNMBase
from rheojax.models.hvnm._kernels import (
    hvnm_ber_rate_constant_interphase,
    hvnm_ber_rate_constant_matrix,
    hvnm_creep_compliance_linear_vec,
    hvnm_guth_gold,
    hvnm_interphase_fraction,
    hvnm_interphase_modulus,
    hvnm_relaxation_modulus_vec,
    hvnm_relaxation_modulus_with_diffusion_vec,
    hvnm_saos_moduli_vec,
    hvnm_startup_stress_linear_vec,
    hvnm_steady_shear_stress_vec,
    hvnm_total_normal_stress_1,
    hvnm_total_stress_shear,
)
from rheojax.models.hvnm._kernels_diffrax import (
    hvnm_solve_creep,
    hvnm_solve_laos,
    hvnm_solve_relaxation,
    hvnm_solve_startup,
)

jax, jnp = safe_import_jax()

logger = logging.getLogger(__name__)


@ModelRegistry.register(
    "hvnm_local",
    protocols=[
        Protocol.FLOW_CURVE,
        Protocol.OSCILLATION,
        Protocol.STARTUP,
        Protocol.RELAXATION,
        Protocol.CREEP,
        Protocol.LAOS,
    ],
    deformation_modes=[
        DeformationMode.SHEAR,
        DeformationMode.TENSION,
        DeformationMode.BENDING,
        DeformationMode.COMPRESSION,
    ],
)
@ModelRegistry.register(
    "hvnm",
    protocols=[
        Protocol.FLOW_CURVE,
        Protocol.OSCILLATION,
        Protocol.STARTUP,
        Protocol.RELAXATION,
        Protocol.CREEP,
        Protocol.LAOS,
    ],
    deformation_modes=[
        DeformationMode.SHEAR,
        DeformationMode.TENSION,
        DeformationMode.BENDING,
        DeformationMode.COMPRESSION,
    ],
)
class HVNMLocal(HVNMBase):
    """Local (0D) Hybrid Vitrimer Nanocomposite Model.

    A constitutive model for NP-filled vitrimers combining:
    - Permanent network (P): covalent crosslinks, amplified by X(phi)
    - Exchangeable network (E): vitrimer bonds with matrix TST kinetics
    - Dissociative network (D): physical bonds, Maxwell relaxation
    - Interphase network (I): NP-bound chains with interfacial TST kinetics

    Parameters
    ----------
    kinetics : {'stress', 'stretch'}, default 'stress'
        TST coupling mechanism for bond exchange rates
    include_damage : bool, default False
        Enable matrix cooperative damage shielding
    include_dissociative : bool, default True
        Include dissociative (D) network
    include_interfacial_damage : bool, default False
        Enable interfacial damage with self-healing
    include_diffusion : bool, default False
        Enable diffusion-limited relaxation tails

    Examples
    --------
    >>> from rheojax.models import HVNMLocal
    >>> model = HVNMLocal()
    >>> model.parameters.set_value("phi", 0.1)
    >>> omega = np.logspace(-2, 2, 50)
    >>> G_prime, G_double_prime = model.predict_saos(omega)

    >>> # Unfilled limit (recovers HVM)
    >>> model = HVNMLocal()
    >>> model.parameters.set_value("phi", 0.0)
    """

    def __init__(
        self,
        kinetics: Literal["stress", "stretch"] = "stress",
        include_damage: bool = False,
        include_dissociative: bool = True,
        include_interfacial_damage: bool = False,
        include_diffusion: bool = False,
    ):
        super().__init__(
            kinetics=kinetics,
            include_damage=include_damage,
            include_dissociative=include_dissociative,
            include_interfacial_damage=include_interfacial_damage,
            include_diffusion=include_diffusion,
        )
        self._setup_parameters()
        self._test_mode = None
        self._gamma_dot_applied = None
        self._sigma_applied = None
        self._gamma_0 = None
        self._omega_laos = None
        logger.info(
            "HVNMLocal initialized",
            extra={
                "kinetics": kinetics,
                "include_damage": include_damage,
                "include_dissociative": include_dissociative,
                "include_interfacial_damage": include_interfacial_damage,
                "include_diffusion": include_diffusion,
                "n_params": len(self.parameters),
            },
        )

    # =========================================================================
    # Parameter Helpers
    # =========================================================================

    def _get_params_dict(self) -> dict[str, float]:
        """Get parameters as dict with defaults for optional params."""
        d = self.get_parameter_dict()
        d.setdefault("G_D", 0.0)
        d.setdefault("k_d_D", 1.0)
        d.setdefault("Gamma_0", 0.0)
        d.setdefault("lambda_crit", 10.0)
        d.setdefault("Gamma_0_int", 0.0)
        d.setdefault("lambda_crit_int", 10.0)
        d.setdefault("h_0", 0.0)
        d.setdefault("E_a_heal", 100e3)
        d.setdefault("n_h", 1.0)
        return d

    def _get_derived_params(self, p_dict: dict) -> dict[str, float]:
        """Compute derived NP geometry quantities from parameter dict.

        Parameters
        ----------
        p_dict : dict
            Parameter name → value mapping

        Returns
        -------
        dict
            Derived quantities: G_I_eff, X_phi, X_I, k_BER_mat_0, k_BER_int_0, phi_I
        """
        phi = p_dict.get("phi", 0.0)
        R_NP = p_dict.get("R_NP", 20e-9)
        delta_m = p_dict.get("delta_m", 10e-9)
        G_E = p_dict.get("G_E", 0.0)
        beta_I = p_dict.get("beta_I", 3.0)
        nu_0 = p_dict.get("nu_0", 1e10)
        E_a = p_dict.get("E_a", 80e3)
        nu_0_int = p_dict.get("nu_0_int", 1e10)
        E_a_int = p_dict.get("E_a_int", 90e3)
        T = p_dict.get("T", 300.0)

        delta_g = 1e-9  # Default glassy layer thickness
        phi_I = hvnm_interphase_fraction(phi, R_NP, delta_g, delta_m)
        G_I_eff = hvnm_interphase_modulus(G_E, beta_I, phi_I)
        X_phi = hvnm_guth_gold(phi)
        # Effective phi for interphase amplification
        from rheojax.models.hvnm._kernels import hvnm_effective_phi
        phi_eff = hvnm_effective_phi(phi, R_NP, delta_g)
        X_I = hvnm_guth_gold(phi_eff)

        k_BER_mat_0 = hvnm_ber_rate_constant_matrix(nu_0, E_a, T)
        k_BER_int_0 = hvnm_ber_rate_constant_interphase(nu_0_int, E_a_int, T)

        return {
            "G_I_eff": G_I_eff,
            "X_phi": X_phi,
            "X_I": X_I,
            "phi_I": phi_I,
            "k_BER_mat_0": k_BER_mat_0,
            "k_BER_int_0": k_BER_int_0,
        }

    def _get_ode_args(self, p_dict: dict | None = None) -> dict:
        """Build complete ODE args dict with derived quantities."""
        if p_dict is None:
            p_dict = self._get_params_dict()
        derived = self._get_derived_params(p_dict)
        args = {**p_dict, **derived}
        # Ensure all interphase params have defaults
        args.setdefault("nu_0_int", 1e10)
        args.setdefault("E_a_int", 90e3)
        args.setdefault("V_act_int", 5e-6)
        args.setdefault("Gamma_0_int", 0.0)
        args.setdefault("lambda_crit_int", 10.0)
        args.setdefault("h_0", 0.0)
        args.setdefault("E_a_heal", 100e3)
        args.setdefault("n_h", 1.0)
        return args

    # =========================================================================
    # Flow Curve (Analytical)
    # =========================================================================

    def predict_flow_curve(
        self, gamma_dot: np.ndarray, return_components: bool = False
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Predict steady-state flow curve.

        At steady state, mu^E -> mu^E_nat and mu^I -> mu^I_nat,
        so sigma_E -> 0 and sigma_I -> 0.
        Only the D-network contributes viscous stress: sigma_D = eta_D * gamma_dot.

        Parameters
        ----------
        gamma_dot : array-like
            Shear rate array (1/s)
        return_components : bool, default False
            If True, return dict with subnetwork contributions

        Returns
        -------
        np.ndarray or dict
            Steady-state stress (Pa) or component dict
        """
        G_D = self.G_D
        k_d_D = self.k_d_D
        assert G_D is not None
        assert k_d_D is not None

        gamma_dot_jax = jnp.asarray(gamma_dot, dtype=jnp.float64)
        sigma = hvnm_steady_shear_stress_vec(
            gamma_dot_jax, G_D, k_d_D
        )

        if return_components:
            eta_D = G_D / max(k_d_D, 1e-30)
            sigma_D = eta_D * gamma_dot_jax
            return {
                "stress": np.asarray(sigma),
                "sigma_P": np.zeros_like(np.asarray(gamma_dot)),
                "sigma_E": np.zeros_like(np.asarray(gamma_dot)),
                "sigma_D": np.asarray(sigma_D),
                "sigma_I": np.zeros_like(np.asarray(gamma_dot)),
                "eta_eff": np.asarray(sigma / jnp.maximum(gamma_dot_jax, 1e-30)),
            }
        return np.asarray(sigma)

    # =========================================================================
    # SAOS (Analytical)
    # =========================================================================

    def predict_saos(
        self, omega: np.ndarray, return_components: bool = True
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Predict SAOS storage and loss moduli.

        Three Maxwell modes (E, D, I) plus amplified permanent plateau (P).

        Parameters
        ----------
        omega : array-like
            Angular frequency array (rad/s)
        return_components : bool, default True
            If True, return (G', G''); if False, return |G*|

        Returns
        -------
        tuple of (np.ndarray, np.ndarray) or np.ndarray
            (G', G'') or |G*|
        """
        G_P = self.G_P
        G_E = self.G_E
        G_D = self.G_D
        k_d_D = self.k_d_D
        assert G_P is not None
        assert G_E is not None
        assert G_D is not None
        assert k_d_D is not None

        omega_jax = jnp.asarray(omega, dtype=jnp.float64)
        p = self._get_params_dict()
        d = self._get_derived_params(p)

        G_prime, G_double_prime = hvnm_saos_moduli_vec(
            omega_jax,
            G_P, G_E, G_D, d["G_I_eff"],
            d["X_phi"], d["X_I"],
            d["k_BER_mat_0"], k_d_D, d["k_BER_int_0"],
            0.0, 0.0,  # D=0, D_int=0
        )

        if return_components:
            return np.asarray(G_prime), np.asarray(G_double_prime)
        return np.asarray(jnp.sqrt(G_prime**2 + G_double_prime**2))

    # =========================================================================
    # Startup Shear
    # =========================================================================

    def simulate_startup(
        self,
        t: np.ndarray,
        gamma_dot: float,
        return_full: bool = False,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Simulate startup shear flow with dual TST feedback.

        Parameters
        ----------
        t : array-like
            Time array (s)
        gamma_dot : float
            Applied shear rate (1/s)
        return_full : bool, default False
            If True, return dict with all trajectories

        Returns
        -------
        np.ndarray or dict
            Stress array or full trajectory dict
        """
        G_P = self.G_P
        G_E = self.G_E
        G_D = self.G_D
        assert G_P is not None
        assert G_E is not None
        assert G_D is not None

        self._gamma_dot_applied = gamma_dot
        t_jax = jnp.asarray(t, dtype=jnp.float64)
        args = self._get_ode_args()

        sol = hvnm_solve_startup(
            t_jax, gamma_dot, args,
            kinetics=self._kinetics,
            include_damage=self._include_damage,
            include_dissociative=self._include_dissociative,
            include_interfacial_damage=self._include_interfacial_damage,
        )

        ys = sol.ys
        assert ys is not None
        D_int_col = jnp.where(
            self._include_interfacial_damage, ys[:, 17], jnp.zeros(len(t_jax))
        ) if self._include_interfacial_damage else jnp.zeros(len(t_jax))

        # Compute total stress from state
        X_phi = args["X_phi"]
        X_I = args["X_I"]
        G_I_eff = args["G_I_eff"]

        stress = jax.vmap(
            lambda y_D_int: hvnm_total_stress_shear(
                y_D_int[0][9],     # gamma
                y_D_int[0][2],     # mu_E_xy
                y_D_int[0][5],     # mu_E_nat_xy
                y_D_int[0][8],     # mu_D_xy
                y_D_int[0][13],    # mu_I_xy
                y_D_int[0][16],    # mu_I_nat_xy
                G_P, G_E, G_D, G_I_eff,
                X_phi, X_I,
                y_D_int[0][10],    # D
                y_D_int[1],        # D_int
            )
        )((ys, D_int_col))

        # Handle solver failure
        stress = jnp.where(
            sol.result == diffrax.RESULTS.successful,
            stress,
            jnp.nan * jnp.ones_like(stress),
        )

        if return_full:
            return {
                "time": np.asarray(t),
                "stress": np.asarray(stress),
                "strain": np.asarray(ys[:, 9]),
                "mu_E_xx": np.asarray(ys[:, 0]),
                "mu_E_yy": np.asarray(ys[:, 1]),
                "mu_E_xy": np.asarray(ys[:, 2]),
                "mu_E_nat_xx": np.asarray(ys[:, 3]),
                "mu_E_nat_yy": np.asarray(ys[:, 4]),
                "mu_E_nat_xy": np.asarray(ys[:, 5]),
                "mu_D_xx": np.asarray(ys[:, 6]),
                "mu_D_yy": np.asarray(ys[:, 7]),
                "mu_D_xy": np.asarray(ys[:, 8]),
                "damage": np.asarray(ys[:, 10]),
                "mu_I_xx": np.asarray(ys[:, 11]),
                "mu_I_yy": np.asarray(ys[:, 12]),
                "mu_I_xy": np.asarray(ys[:, 13]),
                "mu_I_nat_xx": np.asarray(ys[:, 14]),
                "mu_I_nat_yy": np.asarray(ys[:, 15]),
                "mu_I_nat_xy": np.asarray(ys[:, 16]),
                "damage_int": np.asarray(D_int_col),
            }
        return np.asarray(stress)

    # =========================================================================
    # Stress Relaxation
    # =========================================================================

    def simulate_relaxation(
        self,
        t: np.ndarray,
        gamma_step: float = 1.0,
        return_full: bool = False,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Simulate stress relaxation after step strain.

        Parameters
        ----------
        t : array-like
            Time array after step (s)
        gamma_step : float, default 1.0
            Applied step strain
        return_full : bool, default False
            If True, return full trajectory dict

        Returns
        -------
        np.ndarray or dict
            G(t) relaxation modulus or trajectory dict
        """
        G_P = self.G_P
        G_E = self.G_E
        G_D = self.G_D
        assert G_P is not None
        assert G_E is not None
        assert G_D is not None

        t_jax = jnp.asarray(t, dtype=jnp.float64)
        args = self._get_ode_args()

        sol = hvnm_solve_relaxation(
            t_jax, gamma_step, args,
            kinetics=self._kinetics,
            include_damage=self._include_damage,
            include_dissociative=self._include_dissociative,
            include_interfacial_damage=self._include_interfacial_damage,
        )

        ys = sol.ys
        assert ys is not None
        D_int_col = (
            ys[:, 17] if self._include_interfacial_damage
            else jnp.zeros(len(t_jax))
        )
        X_phi = args["X_phi"]
        X_I = args["X_I"]
        G_I_eff = args["G_I_eff"]

        stress = jax.vmap(
            lambda y_D_int: hvnm_total_stress_shear(
                y_D_int[0][9], y_D_int[0][2], y_D_int[0][5], y_D_int[0][8],
                y_D_int[0][13], y_D_int[0][16],
                G_P, G_E, G_D, G_I_eff,
                X_phi, X_I, y_D_int[0][10], y_D_int[1],
            )
        )((ys, D_int_col))

        G_t = stress / jnp.maximum(jnp.abs(gamma_step), 1e-30)
        G_t = jnp.where(
            sol.result == diffrax.RESULTS.successful,
            G_t, jnp.nan * jnp.ones_like(G_t),
        )

        if return_full:
            return {
                "time": np.asarray(t),
                "G_t": np.asarray(G_t),
                "stress": np.asarray(stress),
                "mu_E_xy": np.asarray(ys[:, 2]),
                "mu_E_nat_xy": np.asarray(ys[:, 5]),
                "mu_D_xy": np.asarray(ys[:, 8]),
                "mu_I_xy": np.asarray(ys[:, 13]),
                "mu_I_nat_xy": np.asarray(ys[:, 16]),
                "damage": np.asarray(ys[:, 10]),
                "damage_int": np.asarray(D_int_col),
            }
        return np.asarray(G_t)

    # =========================================================================
    # Creep
    # =========================================================================

    def simulate_creep(
        self,
        t: np.ndarray,
        sigma_0: float,
        return_full: bool = False,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Simulate creep under constant stress.

        Parameters
        ----------
        t : array-like
            Time array (s)
        sigma_0 : float
            Applied constant stress (Pa)
        return_full : bool, default False
            If True, return full trajectory dict

        Returns
        -------
        np.ndarray or dict
            Strain gamma(t) or trajectory dict
        """
        self._sigma_applied = sigma_0
        t_jax = jnp.asarray(t, dtype=jnp.float64)
        args = self._get_ode_args()
        assert args is not None

        sol = hvnm_solve_creep(
            t_jax, sigma_0, args,
            kinetics=self._kinetics,
            include_damage=self._include_damage,
            include_dissociative=self._include_dissociative,
            include_interfacial_damage=self._include_interfacial_damage,
        )

        ys = sol.ys
        assert ys is not None
        gamma = ys[:, 9]
        gamma = jnp.where(
            sol.result == diffrax.RESULTS.successful,
            gamma, jnp.nan * jnp.ones_like(gamma),
        )

        if return_full:
            J_t = gamma / jnp.maximum(jnp.abs(sigma_0), 1e-30)
            D_int_col = (
                ys[:, 17] if self._include_interfacial_damage
                else jnp.zeros(len(t_jax))
            )
            return {
                "time": np.asarray(t),
                "strain": np.asarray(gamma),
                "compliance": np.asarray(J_t),
                "mu_E_xy": np.asarray(ys[:, 2]),
                "mu_E_nat_xy": np.asarray(ys[:, 5]),
                "mu_D_xy": np.asarray(ys[:, 8]),
                "mu_I_xy": np.asarray(ys[:, 13]),
                "mu_I_nat_xy": np.asarray(ys[:, 16]),
                "damage": np.asarray(ys[:, 10]),
                "damage_int": np.asarray(D_int_col),
            }
        return np.asarray(gamma)

    # =========================================================================
    # LAOS
    # =========================================================================

    def simulate_laos(
        self,
        t: np.ndarray,
        gamma_0: float,
        omega: float,
    ) -> dict[str, np.ndarray]:
        """Simulate LAOS (Large Amplitude Oscillatory Shear).

        Parameters
        ----------
        t : array-like
            Time array (s)
        gamma_0 : float
            Strain amplitude
        omega : float
            Angular frequency (rad/s)

        Returns
        -------
        dict
            Keys: time, strain, stress, gamma_dot, N1,
            mu_E_xy, mu_E_nat_xy, mu_D_xy, mu_I_xy, mu_I_nat_xy,
            damage, damage_int
        """
        G_P = self.G_P
        G_E = self.G_E
        G_D = self.G_D
        assert G_P is not None
        assert G_E is not None
        assert G_D is not None

        self._gamma_0 = gamma_0
        self._omega_laos = omega
        t_jax = jnp.asarray(t, dtype=jnp.float64)
        args = self._get_ode_args()

        sol = hvnm_solve_laos(
            t_jax, gamma_0, omega, args,
            kinetics=self._kinetics,
            include_damage=self._include_damage,
            include_dissociative=self._include_dissociative,
            include_interfacial_damage=self._include_interfacial_damage,
        )

        ys = sol.ys
        assert ys is not None
        D_int_col = (
            ys[:, 17] if self._include_interfacial_damage
            else jnp.zeros(len(t_jax))
        )
        strain = gamma_0 * jnp.sin(omega * t_jax)
        gamma_dot_arr = gamma_0 * omega * jnp.cos(omega * t_jax)
        X_phi = args["X_phi"]
        X_I = args["X_I"]
        G_I_eff = args["G_I_eff"]

        stress = jax.vmap(
            lambda y_D_int: hvnm_total_stress_shear(
                y_D_int[0][9], y_D_int[0][2], y_D_int[0][5], y_D_int[0][8],
                y_D_int[0][13], y_D_int[0][16],
                G_P, G_E, G_D, G_I_eff,
                X_phi, X_I, y_D_int[0][10], y_D_int[1],
            )
        )((ys, D_int_col))

        N1 = jax.vmap(
            lambda y_D_int: hvnm_total_normal_stress_1(
                y_D_int[0][0], y_D_int[0][1], y_D_int[0][3], y_D_int[0][4],
                y_D_int[0][6], y_D_int[0][7],
                y_D_int[0][11], y_D_int[0][12], y_D_int[0][14], y_D_int[0][15],
                G_E, G_D, G_I_eff, X_I, y_D_int[1],
            )
        )((ys, D_int_col))

        failed = sol.result != diffrax.RESULTS.successful
        stress = jnp.where(failed, jnp.nan, stress)
        N1 = jnp.where(failed, jnp.nan, N1)

        return {
            "time": np.asarray(t),
            "strain": np.asarray(strain),
            "stress": np.asarray(stress),
            "gamma_dot": np.asarray(gamma_dot_arr),
            "N1": np.asarray(N1),
            "mu_E_xy": np.asarray(ys[:, 2]),
            "mu_E_nat_xy": np.asarray(ys[:, 5]),
            "mu_D_xy": np.asarray(ys[:, 8]),
            "mu_I_xy": np.asarray(ys[:, 13]),
            "mu_I_nat_xy": np.asarray(ys[:, 16]),
            "damage": np.asarray(ys[:, 10]),
            "damage_int": np.asarray(D_int_col),
        }

    # =========================================================================
    # Normal Stresses
    # =========================================================================

    def predict_normal_stresses(
        self, gamma_dot: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict steady-state normal stress differences.

        At steady state, E and I networks contribute zero normal stress.
        Only D-network contributes N1.

        N1 = 2 * G_D * (gamma_dot / k_d_D)^2
        N2 = 0

        Parameters
        ----------
        gamma_dot : array-like
            Shear rate array (1/s)

        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            (N1, N2) arrays (Pa)
        """
        G_D = self.G_D
        k_d_D = self.k_d_D
        assert G_D is not None
        assert k_d_D is not None

        gamma_dot_jax = jnp.asarray(gamma_dot, dtype=jnp.float64)
        Wi_D = gamma_dot_jax / max(k_d_D, 1e-30)
        N1 = 2.0 * G_D * Wi_D**2
        N2 = jnp.zeros_like(N1)
        return np.asarray(N1), np.asarray(N2)

    # =========================================================================
    # LAOS Harmonic Extraction
    # =========================================================================

    def extract_laos_harmonics(
        self, laos_result: dict[str, np.ndarray], n_harmonics: int = 5
    ) -> dict[str, np.ndarray]:
        """Extract Fourier harmonics from LAOS simulation.

        Parameters
        ----------
        laos_result : dict
            Output from simulate_laos()
        n_harmonics : int, default 5
            Number of harmonics to extract

        Returns
        -------
        dict
            Keys: harmonic_index (1, 3, 5, ...), sigma_harmonics
        """
        t = laos_result["time"]
        stress = laos_result["stress"]
        omega = self._omega_laos or 1.0

        period = 2.0 * np.pi / omega
        n_periods = int((t[-1] - t[0]) / period)
        if n_periods < 1:
            n_periods = 1
        t_start = t[-1] - n_periods * period
        mask = t >= t_start

        t_cycle = t[mask]
        stress_cycle = stress[mask]

        n_pts = len(t_cycle)
        dt = (t_cycle[-1] - t_cycle[0]) / max(n_pts - 1, 1)
        freqs = np.fft.rfftfreq(n_pts, d=dt)
        fft_stress = np.fft.rfft(stress_cycle)

        f_fundamental = omega / (2.0 * np.pi)
        harmonics_idx = np.arange(1, 2 * n_harmonics, 2)
        amplitudes = np.zeros(n_harmonics)

        for i, n in enumerate(harmonics_idx):
            target_freq = n * f_fundamental
            idx = np.argmin(np.abs(freqs - target_freq))
            amplitudes[i] = 2.0 * np.abs(fft_stress[idx]) / n_pts

        return {
            "harmonic_index": harmonics_idx,
            "sigma_harmonics": amplitudes,
        }

    # =========================================================================
    # Payne Effect Parameters
    # =========================================================================

    def get_payne_parameters(self) -> dict[str, float]:
        """Extract Payne effect parameters from model.

        The Payne effect manifests as modulus drop with increasing
        strain amplitude, driven by interphase softening.

        Returns
        -------
        dict
            G_0: zero-strain modulus
            G_inf: high-strain modulus (X*G_P only)
            gamma_c: approximate critical strain (1/X_I)
        """
        G_P = self.G_P
        G_E = self.G_E
        G_D = self.G_D
        assert G_P is not None
        assert G_E is not None
        assert G_D is not None

        d = self._get_derived_params(self._get_params_dict())
        G_I_amp = d["G_I_eff"] * d["X_I"]
        G_0 = G_P * d["X_phi"] + G_E + G_D + G_I_amp
        G_inf = G_P * d["X_phi"]  # Only permanent plateau at large strain
        gamma_c = 1.0 / max(d["X_I"], 1.0)  # Critical strain ~ 1/X_I
        return {"G_0": float(G_0), "G_inf": float(G_inf), "gamma_c": float(gamma_c)}

    # =========================================================================
    # Fitting (NLSQ)
    # =========================================================================

    def _fit(self, x, y, **kwargs):
        """Fit model to data using protocol-aware optimization.

        Parameters
        ----------
        x : array-like
            Independent variable
        y : array-like
            Dependent variable
        **kwargs
            test_mode, gamma_dot, sigma_applied, gamma_0, omega, etc.

        Returns
        -------
        self
        """
        from rheojax.utils.optimization import (
            create_least_squares_objective,
            nlsq_optimize,
        )

        test_mode = kwargs.get("test_mode", self._test_mode or "flow_curve")
        self._test_mode = test_mode

        x_jax = jnp.asarray(x, dtype=jnp.float64)
        y_jax = jnp.asarray(y, dtype=jnp.float64)

        self._gamma_dot_applied = kwargs.get("gamma_dot")
        self._sigma_applied = kwargs.get("sigma_applied")
        self._gamma_0 = kwargs.get("gamma_0")
        self._omega_laos = kwargs.get("omega")

        # Filter out fitting-specific and BaseModel kwargs
        fwd_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in (
                "test_mode", "deformation_mode", "poisson_ratio",
                "use_log_residuals", "use_jax", "method",
                "max_iter", "use_multi_start", "n_starts", "perturb_factor",
            )
        }

        def model_fn(x_fit, params):
            return self.model_function(x_fit, params, test_mode=test_mode, **fwd_kwargs)

        objective = create_least_squares_objective(
            model_fn,
            x_jax,
            y_jax,
            use_log_residuals=kwargs.get(
                "use_log_residuals", test_mode == "flow_curve"
            ),
        )

        result = nlsq_optimize(
            objective,
            self.parameters,
            use_jax=kwargs.get("use_jax", True),
            method=kwargs.get("method", "auto"),
            max_iter=kwargs.get("max_iter", 2000),
        )

        self.fitted_ = True
        self._nlsq_result = result

        logger.info(
            f"Fitted HVNMLocal: G_P={self.G_P:.2e}, G_E={self.G_E:.2e}, "
            f"phi={self.phi:.3f}"
        )

        return self

    def _predict(self, X, **kwargs):
        """Predict response using fitted parameters."""
        test_mode = kwargs.get("test_mode", self._test_mode or "flow_curve")
        param_values = jnp.array(
            [self.parameters.get_value(n) for n in self.parameters.keys()],
            dtype=jnp.float64,
        )
        fwd_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ("test_mode", "deformation_mode", "poisson_ratio")
        }
        return np.asarray(
            self.model_function(X, param_values, test_mode=test_mode, **fwd_kwargs)
        )

    # =========================================================================
    # Model Function (NLSQ / NumPyro)
    # =========================================================================

    def model_function(self, X, params, test_mode=None, **kwargs):
        """NumPyro/BayesianMixin model function for HVNM.

        Routes to appropriate JAX-traceable prediction based on test_mode.
        Required by BayesianMixin for NumPyro NUTS sampling.

        Parameters
        ----------
        X : array-like
            Independent variable
        params : array-like
            Parameter values in ParameterSet order
        test_mode : str, optional
            Override stored test mode
        **kwargs
            Protocol-specific: gamma_dot, sigma_applied, gamma_0, omega

        Returns
        -------
        jnp.ndarray
            Predicted response
        """
        # Unpack parameters by position
        p_names = list(self.parameters.keys())
        p_dict = dict(zip(p_names, params, strict=True))

        G_P = p_dict["G_P"]
        G_E = p_dict["G_E"]
        nu_0 = p_dict["nu_0"]
        E_a = p_dict["E_a"]
        V_act = p_dict["V_act"]
        T = p_dict["T"]
        G_D = p_dict.get("G_D", 0.0)
        k_d_D = p_dict.get("k_d_D", 1.0)

        # Interphase params
        beta_I = p_dict.get("beta_I", 3.0)
        nu_0_int = p_dict.get("nu_0_int", 1e10)
        E_a_int = p_dict.get("E_a_int", 90e3)
        V_act_int = p_dict.get("V_act_int", 5e-6)
        phi = p_dict.get("phi", 0.0)
        R_NP = p_dict.get("R_NP", 20e-9)
        delta_m = p_dict.get("delta_m", 10e-9)

        mode = test_mode or self._test_mode or "flow_curve"
        X_jax = jnp.asarray(X, dtype=jnp.float64)

        gamma_dot = kwargs.get("gamma_dot", self._gamma_dot_applied)
        sigma_applied = kwargs.get("sigma_applied", self._sigma_applied)
        gamma_0 = kwargs.get("gamma_0", self._gamma_0)
        omega = kwargs.get("omega", self._omega_laos)

        # Compute derived quantities (JAX-traceable)
        delta_g = 1e-9
        phi_I = hvnm_interphase_fraction(phi, R_NP, delta_g, delta_m)
        G_I_eff = hvnm_interphase_modulus(G_E, beta_I, phi_I)
        X_phi = hvnm_guth_gold(phi)
        from rheojax.models.hvnm._kernels import hvnm_effective_phi
        phi_eff = hvnm_effective_phi(phi, R_NP, delta_g)
        X_I = hvnm_guth_gold(phi_eff)
        k_BER_mat_0 = hvnm_ber_rate_constant_matrix(nu_0, E_a, T)
        k_BER_int_0 = hvnm_ber_rate_constant_interphase(nu_0_int, E_a_int, T)

        if mode in ["flow_curve", "steady_shear", "rotation"]:
            return hvnm_steady_shear_stress_vec(X_jax, G_D, k_d_D)

        elif mode == "oscillation":
            G_prime, G_double_prime = hvnm_saos_moduli_vec(
                X_jax, G_P, G_E, G_D, G_I_eff,
                X_phi, X_I,
                k_BER_mat_0, k_d_D, k_BER_int_0,
                0.0, 0.0,  # D=0, D_int=0
            )
            return jnp.sqrt(G_prime**2 + G_double_prime**2)

        elif mode == "startup":
            if gamma_dot is None:
                raise ValueError("startup mode requires gamma_dot")
            return hvnm_startup_stress_linear_vec(
                X_jax, gamma_dot,
                G_P, G_E, G_D, G_I_eff,
                X_phi, X_I,
                k_BER_mat_0, k_d_D, k_BER_int_0,
                0.0,  # D_int=0
            )

        elif mode == "relaxation":
            if self._include_diffusion:
                k_diff_mat = p_dict.get("k_diff_0_mat", 0.0)
                k_diff_int = p_dict.get("k_diff_0_int", 0.0)
                return hvnm_relaxation_modulus_with_diffusion_vec(
                    X_jax,
                    G_P, G_E, G_D, G_I_eff,
                    X_phi, X_I,
                    k_BER_mat_0, k_d_D, k_BER_int_0,
                    k_diff_mat, k_diff_int,
                    0.0, 0.0,  # D=0, D_int=0
                )
            return hvnm_relaxation_modulus_vec(
                X_jax,
                G_P, G_E, G_D, G_I_eff,
                X_phi, X_I,
                k_BER_mat_0, k_d_D, k_BER_int_0,
                0.0, 0.0,  # D=0, D_int=0
            )

        elif mode == "creep":
            if sigma_applied is None:
                raise ValueError("creep mode requires sigma_applied")
            J = hvnm_creep_compliance_linear_vec(
                X_jax,
                G_P, G_E, G_D, G_I_eff,
                X_phi, X_I,
                k_BER_mat_0, k_d_D, k_BER_int_0,
            )
            return sigma_applied * J

        elif mode == "laos":
            if gamma_0 is None or omega is None:
                raise ValueError("LAOS mode requires gamma_0 and omega")
            params_dict = {
                "G_P": G_P, "G_E": G_E, "G_D": G_D, "k_d_D": k_d_D,
                "nu_0": nu_0, "E_a": E_a, "V_act": V_act, "T": T,
                "G_I_eff": G_I_eff, "X_phi": X_phi, "X_I": X_I,
                "nu_0_int": nu_0_int, "E_a_int": E_a_int, "V_act_int": V_act_int,
                "Gamma_0": 0.0, "lambda_crit": 10.0,
                "Gamma_0_int": 0.0, "lambda_crit_int": 10.0,
                "h_0": 0.0, "E_a_heal": 100e3, "n_h": 1.0,
            }
            sol = hvnm_solve_laos(
                X_jax, gamma_0, omega, params_dict,
                kinetics=self._kinetics,
                include_damage=False,
                include_dissociative=self._include_dissociative,
                include_interfacial_damage=False,
            )
            stress = jax.vmap(
                lambda y: hvnm_total_stress_shear(
                    y[9], y[2], y[5], y[8], y[13], y[16],
                    G_P, G_E, G_D, G_I_eff, X_phi, X_I, y[10], 0.0,
                )
            )(sol.ys)
            return stress

        else:
            logger.warning(f"Unknown test_mode '{mode}', defaulting to flow_curve")
            return hvnm_steady_shear_stress_vec(X_jax, G_D, k_d_D)

    # =========================================================================
    # Factory Methods (Limiting Cases)
    # =========================================================================

    @classmethod
    def unfilled_vitrimer(
        cls,
        G_P: float = 1e4,
        G_E: float = 1e4,
        G_D: float = 1e3,
        nu_0: float = 1e10,
        E_a: float = 80e3,
        V_act: float = 1e-5,
        T: float = 300.0,
        k_d_D: float = 1.0,
    ) -> HVNMLocal:
        """Create unfilled vitrimer (phi=0, recovers HVM exactly).

        Parameters
        ----------
        G_P, G_E, G_D : float
            Subnetwork moduli (Pa)
        nu_0, E_a, V_act, T : float
            TST parameters
        k_d_D : float
            Dissociative rate (1/s)

        Returns
        -------
        HVNMLocal
            Model with phi=0 (no interphase contribution)
        """
        model = cls(include_dissociative=True)
        model.parameters.set_value("G_P", G_P)
        model.parameters.set_value("G_E", G_E)
        model.parameters.set_value("G_D", G_D)
        model.parameters.set_value("nu_0", nu_0)
        model.parameters.set_value("E_a", E_a)
        model.parameters.set_value("V_act", V_act)
        model.parameters.set_value("T", T)
        model.parameters.set_value("k_d_D", k_d_D)
        model.parameters.set_value("phi", 0.0)
        return model

    @classmethod
    def filled_elastomer(
        cls,
        G_P: float = 1e4,
        phi: float = 0.1,
        R_NP: float = 20e-9,
        delta_m: float = 10e-9,
    ) -> HVNMLocal:
        """Create filled elastomer (no exchange networks).

        Parameters
        ----------
        G_P : float
            Permanent network modulus (Pa)
        phi : float
            NP volume fraction
        R_NP : float
            NP radius (m)
        delta_m : float
            Mobile interphase thickness (m)

        Returns
        -------
        HVNMLocal
            Model with only amplified P-network (no E, D, or active I)
        """
        model = cls(include_dissociative=False)
        model.parameters.set_value("G_P", G_P)
        model.parameters.set_value("G_E", 0.0)
        model.parameters.set_value("phi", phi)
        model.parameters.set_value("R_NP", R_NP)
        model.parameters.set_value("delta_m", delta_m)
        return model

    @classmethod
    def partial_vitrimer_nc(
        cls,
        G_P: float = 1e4,
        G_E: float = 1e4,
        phi: float = 0.1,
        nu_0: float = 1e10,
        E_a: float = 80e3,
        V_act: float = 1e-5,
        T: float = 300.0,
        **nc_kwargs,
    ) -> HVNMLocal:
        """Create partial vitrimer nanocomposite (G_D=0).

        Parameters
        ----------
        G_P, G_E : float
            Network moduli (Pa)
        phi : float
            NP volume fraction
        nu_0, E_a, V_act, T : float
            TST parameters
        **nc_kwargs
            NP geometry: R_NP, delta_m, beta_I, nu_0_int, E_a_int, V_act_int

        Returns
        -------
        HVNMLocal
            Model with P + E + I networks (no D)
        """
        model = cls(include_dissociative=False)
        model.parameters.set_value("G_P", G_P)
        model.parameters.set_value("G_E", G_E)
        model.parameters.set_value("phi", phi)
        model.parameters.set_value("nu_0", nu_0)
        model.parameters.set_value("E_a", E_a)
        model.parameters.set_value("V_act", V_act)
        model.parameters.set_value("T", T)
        for key, val in nc_kwargs.items():
            if key in model.parameters.keys():
                model.parameters.set_value(key, val)
        return model

    @classmethod
    def conventional_filled_rubber(
        cls,
        G_P: float = 1e4,
        phi: float = 0.1,
        R_NP: float = 20e-9,
        delta_m: float = 10e-9,
        G_D: float = 1e3,
        k_d_D: float = 1.0,
    ) -> HVNMLocal:
        """Create conventional filled rubber (no E-network, frozen interphase).

        Parameters
        ----------
        G_P : float
            Permanent network modulus (Pa)
        phi : float
            NP volume fraction
        R_NP, delta_m : float
            NP geometry (m)
        G_D : float
            Dissociative modulus (Pa)
        k_d_D : float
            Dissociative rate (1/s)

        Returns
        -------
        HVNMLocal
            Model with P + D + frozen I (no exchange)
        """
        model = cls(include_dissociative=True)
        model.parameters.set_value("G_P", G_P)
        model.parameters.set_value("G_E", 0.0)
        model.parameters.set_value("G_D", G_D)
        model.parameters.set_value("k_d_D", k_d_D)
        model.parameters.set_value("phi", phi)
        model.parameters.set_value("R_NP", R_NP)
        model.parameters.set_value("delta_m", delta_m)
        return model

    @classmethod
    def matrix_only_exchange(
        cls,
        G_P: float = 1e4,
        G_E: float = 1e4,
        phi: float = 0.1,
        nu_0: float = 1e10,
        E_a: float = 80e3,
        V_act: float = 1e-5,
        T: float = 300.0,
        **nc_kwargs,
    ) -> HVNMLocal:
        """Create model with frozen interphase (k_BER^int=0).

        The interphase acts as a dead (non-exchanging) reinforcement layer.

        Parameters
        ----------
        G_P, G_E : float
            Network moduli (Pa)
        phi : float
            NP volume fraction
        nu_0, E_a, V_act, T : float
            Matrix TST parameters
        **nc_kwargs
            NP geometry: R_NP, delta_m, beta_I

        Returns
        -------
        HVNMLocal
            Model with active matrix exchange, frozen interphase
        """
        model = cls(include_dissociative=True)
        model.parameters.set_value("G_P", G_P)
        model.parameters.set_value("G_E", G_E)
        model.parameters.set_value("phi", phi)
        model.parameters.set_value("nu_0", nu_0)
        model.parameters.set_value("E_a", E_a)
        model.parameters.set_value("V_act", V_act)
        model.parameters.set_value("T", T)
        # Freeze interphase: max barrier within bounds → k_BER^int ≈ 0
        model.parameters.set_value("E_a_int", 250e3)  # Max allowed → negligible rate
        for key, val in nc_kwargs.items():
            if key in model.parameters.keys():
                model.parameters.set_value(key, val)
        return model
