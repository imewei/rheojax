"""Base class for HVNM (Hybrid Vitrimer Nanocomposite Model).

Provides shared infrastructure for HVNM variants:
- Conditional parameter registration (HVM base + interphase + optional features)
- NP geometry calculations (Guth-Gold, interphase fraction)
- Dual TST kinetics utilities
- Inherits HVMBase for P, E, D subnetwork infrastructure

The HVNM models a nanoparticle-filled vitrimer with four subnetworks:
1. Permanent (P): covalent crosslinks, amplified by X(phi)
2. Exchangeable (E): associative vitrimer bonds with matrix BER kinetics
3. Dissociative (D): physical reversible bonds, standard Maxwell
4. Interphase (I): NP-bound chains with distinct interfacial BER kinetics

References
----------
- Li, Zhao, Duan, Zhang, Liu (2024). Langmuir 40, 7550-7560.
- Karim, Vernerey, Sain (2025). Macromolecules 58, 4899-4912.
- Papon, Montes et al. (2012). Soft Matter 8, 4090-4096.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from rheojax.core.jax_config import safe_import_jax
from rheojax.core.parameters import ParameterSet
from rheojax.models.hvm._base import HVMBase
from rheojax.models.hvnm._kernels import (
    _R_GAS,
    hvnm_ber_rate_constant_interphase,
    hvnm_ber_rate_constant_matrix,
    hvnm_guth_gold,
    hvnm_interphase_fraction,
    hvnm_interphase_modulus,
)

jax, jnp = safe_import_jax()

logger = logging.getLogger(__name__)


class HVNMBase(HVMBase):
    """Base class for Hybrid Vitrimer Nanocomposite Models.

    Extends HVMBase with:
    - Interphase (I) subnetwork with distinct TST kinetics
    - NP geometry: Guth-Gold amplification, interphase fraction
    - Dual BER rates (matrix and interfacial, independent)
    - Optional interfacial damage with self-healing
    - Optional diffusion mode for long-time tails

    Parameters
    ----------
    kinetics : {'stress', 'stretch'}, default 'stress'
        TST coupling mechanism
    include_damage : bool, default False
        Enable matrix cooperative damage shielding
    include_dissociative : bool, default True
        Include dissociative (D) network
    include_interfacial_damage : bool, default False
        Enable interfacial damage with self-healing
    include_diffusion : bool, default False
        Enable diffusion-limited relaxation tails
    """

    def __init__(
        self,
        kinetics: Literal["stress", "stretch"] = "stress",
        include_damage: bool = False,
        include_dissociative: bool = True,
        include_interfacial_damage: bool = False,
        include_diffusion: bool = False,
    ):
        self._include_interfacial_damage = include_interfacial_damage
        self._include_diffusion = include_diffusion

        super().__init__(
            kinetics=kinetics,
            include_damage=include_damage,
            include_dissociative=include_dissociative,
        )

    # =========================================================================
    # Properties — Interphase
    # =========================================================================

    @property
    def include_interfacial_damage(self) -> bool:
        """Whether interfacial damage is active."""
        return self._include_interfacial_damage

    @property
    def include_diffusion(self) -> bool:
        """Whether diffusion mode is active."""
        return self._include_diffusion

    @property
    def beta_I(self) -> float:
        """Interphase reinforcement ratio G_I/G_E."""
        val = self.parameters.get_value("beta_I")
        return float(val) if val is not None else 3.0

    @property
    def nu_0_int(self) -> float:
        """Interfacial TST attempt frequency (1/s)."""
        val = self.parameters.get_value("nu_0_int")
        return float(val) if val is not None else 1e10

    @property
    def E_a_int(self) -> float:
        """Interfacial activation energy (J/mol)."""
        val = self.parameters.get_value("E_a_int")
        return float(val) if val is not None else 90e3

    @property
    def V_act_int(self) -> float:
        """Interfacial activation volume (m^3/mol)."""
        val = self.parameters.get_value("V_act_int")
        return float(val) if val is not None else 5e-6

    @property
    def phi(self) -> float:
        """NP volume fraction."""
        val = self.parameters.get_value("phi")
        return float(val) if val is not None else 0.05

    @property
    def R_NP(self) -> float:
        """NP radius (m)."""
        val = self.parameters.get_value("R_NP")
        return float(val) if val is not None else 20e-9

    @property
    def delta_m(self) -> float:
        """Mobile interphase thickness (m)."""
        val = self.parameters.get_value("delta_m")
        return float(val) if val is not None else 10e-9

    # --- Interfacial damage properties ---

    @property
    def Gamma_0_int(self) -> float:
        """Interfacial damage rate (1/s)."""
        if not self._include_interfacial_damage:
            return 0.0
        val = self.parameters.get_value("Gamma_0_int")
        return float(val) if val is not None else 0.0

    @property
    def lambda_crit_int(self) -> float:
        """Interfacial critical stretch."""
        if not self._include_interfacial_damage:
            return 10.0
        val = self.parameters.get_value("lambda_crit_int")
        return float(val) if val is not None else 1.5

    @property
    def h_0(self) -> float:
        """Self-healing pre-exponential (1/s)."""
        if not self._include_interfacial_damage:
            return 0.0
        val = self.parameters.get_value("h_0")
        return float(val) if val is not None else 0.0

    @property
    def E_a_heal(self) -> float:
        """Healing activation energy (J/mol)."""
        if not self._include_interfacial_damage:
            return 100e3
        val = self.parameters.get_value("E_a_heal")
        return float(val) if val is not None else 100e3

    @property
    def n_h(self) -> float:
        """Healing exponent."""
        if not self._include_interfacial_damage:
            return 1.0
        val = self.parameters.get_value("n_h")
        return float(val) if val is not None else 1.0

    # --- Diffusion properties ---

    @property
    def k_diff_0_mat(self) -> float:
        """Matrix diffusion rate (1/s)."""
        if not self._include_diffusion:
            return 0.0
        val = self.parameters.get_value("k_diff_0_mat")
        return float(val) if val is not None else 0.0

    @property
    def k_diff_0_int(self) -> float:
        """Interfacial diffusion rate (1/s)."""
        if not self._include_diffusion:
            return 0.0
        val = self.parameters.get_value("k_diff_0_int")
        return float(val) if val is not None else 0.0

    @property
    def E_a_diff(self) -> float:
        """Diffusion activation energy (J/mol)."""
        if not self._include_diffusion:
            return 120e3
        val = self.parameters.get_value("E_a_diff")
        return float(val) if val is not None else 120e3

    # =========================================================================
    # Derived Quantities
    # =========================================================================

    @property
    def phi_I(self) -> float:
        """Interphase volume fraction (from NP geometry)."""
        # Use delta_g = 1e-9 as default glassy layer thickness
        return float(hvnm_interphase_fraction(self.phi, self.R_NP, 1e-9, self.delta_m))

    @property
    def phi_eff(self) -> float:
        """Effective NP volume fraction (with glassy layer)."""
        from rheojax.models.hvnm._kernels import hvnm_effective_phi
        return float(hvnm_effective_phi(self.phi, self.R_NP, 1e-9))

    @property
    def X_phi(self) -> float:
        """Guth-Gold strain amplification X(phi) for P-network."""
        return float(hvnm_guth_gold(self.phi))

    @property
    def X_I(self) -> float:
        """Strain amplification for I-network X(phi_eff)."""
        return float(hvnm_guth_gold(self.phi_eff))

    @property
    def G_I_eff(self) -> float:
        """Effective interphase modulus G_I = beta_I * G_E * phi_I (Pa)."""
        return float(hvnm_interphase_modulus(self.G_E, self.beta_I, self.phi_I))

    # =========================================================================
    # Parameter Setup
    # =========================================================================

    def _setup_parameters(self):
        """Initialize ParameterSet with conditional parameters.

        Always present from HVM (6): G_P, G_E, nu_0, E_a, V_act, T
        Interphase core (7): beta_I, nu_0_int, E_a_int, V_act_int, phi, R_NP, delta_m
        When include_dissociative (+2): G_D, k_d_D
        When include_damage (+2): Gamma_0, lambda_crit
        When include_interfacial_damage (+5): Gamma_0_int, lambda_crit_int, h_0, E_a_heal, n_h
        When include_diffusion (+3): k_diff_0_mat, k_diff_0_int, E_a_diff
        """
        # HVM base parameters (P + E + TST + optional D + optional damage)
        super()._setup_parameters()

        # --- Interphase core parameters ---
        self.parameters.add(
            name="beta_I",
            value=3.0,
            bounds=(1.0, 10.0),
            units="",
            description="Interphase reinforcement ratio G_I/G_E",
        )
        self.parameters.add(
            name="nu_0_int",
            value=1e10,
            bounds=(1e6, 1e14),
            units="1/s",
            description="Interfacial TST attempt frequency",
        )
        self.parameters.add(
            name="E_a_int",
            value=90e3,
            bounds=(30e3, 250e3),
            units="J/mol",
            description="Interfacial activation energy",
        )
        self.parameters.add(
            name="V_act_int",
            value=5e-6,
            bounds=(1e-8, 1e-2),
            units="m^3/mol",
            description="Interfacial activation volume",
        )
        self.parameters.add(
            name="phi",
            value=0.05,
            bounds=(0.0, 0.5),
            units="",
            description="NP volume fraction",
        )
        self.parameters.add(
            name="R_NP",
            value=20e-9,
            bounds=(1e-9, 1e-6),
            units="m",
            description="NP radius",
        )
        self.parameters.add(
            name="delta_m",
            value=10e-9,
            bounds=(1e-9, 100e-9),
            units="m",
            description="Mobile interphase thickness",
        )

        # --- Optional: Interfacial damage ---
        if self._include_interfacial_damage:
            self.parameters.add(
                name="Gamma_0_int",
                value=1e-3,
                bounds=(0.0, 1.0),
                units="1/s",
                description="Interfacial damage rate",
            )
            self.parameters.add(
                name="lambda_crit_int",
                value=1.5,
                bounds=(1.001, 5.0),
                units="",
                description="Interfacial critical stretch",
            )
            self.parameters.add(
                name="h_0",
                value=1e-4,
                bounds=(0.0, 1.0),
                units="1/s",
                description="Self-healing pre-exponential",
            )
            self.parameters.add(
                name="E_a_heal",
                value=100e3,
                bounds=(30e3, 300e3),
                units="J/mol",
                description="Healing activation energy",
            )
            self.parameters.add(
                name="n_h",
                value=1.0,
                bounds=(0.5, 2.0),
                units="",
                description="Healing exponent",
            )

        # --- Optional: Diffusion ---
        if self._include_diffusion:
            self.parameters.add(
                name="k_diff_0_mat",
                value=1e-4,
                bounds=(0.0, 1.0),
                units="1/s",
                description="Matrix diffusion rate",
            )
            self.parameters.add(
                name="k_diff_0_int",
                value=1e-6,
                bounds=(0.0, 0.1),
                units="1/s",
                description="Interfacial diffusion rate",
            )
            self.parameters.add(
                name="E_a_diff",
                value=120e3,
                bounds=(50e3, 400e3),
                units="J/mol",
                description="Diffusion activation energy",
            )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def compute_interphase_fraction(self) -> float:
        """Compute mobile interphase volume fraction from NP geometry.

        Returns
        -------
        float
            phi_I — interphase volume fraction
        """
        return self.phi_I

    def compute_strain_amplification(self) -> float:
        """Compute Guth-Gold strain amplification factor.

        X(phi) = 1 + 2.5*phi + 14.1*phi^2

        Returns
        -------
        float
            Strain amplification factor
        """
        return self.X_phi

    def compute_ber_rate_interphase_equilibrium(self) -> float:
        """Compute thermal interfacial BER rate at zero stress.

        k_BER^int_0 = nu_0_int * exp(-E_a_int / (R*T))

        Returns
        -------
        float
            Zero-stress interfacial BER rate (1/s)
        """
        return float(
            hvnm_ber_rate_constant_interphase(self.nu_0_int, self.E_a_int, self.T)
        )

    def get_interphase_relaxation_time(self) -> float:
        """Effective relaxation time of the interphase network.

        tau_I_eff = 1 / (2 * k_BER^int_0)

        The factor-of-2 arises from the dual natural-state evolution,
        same mechanism as the matrix E-network.

        Returns
        -------
        float
            Effective I-network relaxation time (s)
        """
        k0_int = self.compute_ber_rate_interphase_equilibrium()
        return 1.0 / (2.0 * max(k0_int, 1e-30))

    def get_network_fractions_nc(self) -> dict[str, float]:
        """Compute modulus fractions for all four subnetworks.

        Returns
        -------
        dict
            Keys: 'f_P', 'f_E', 'f_D', 'f_I' with values in [0, 1]
        """
        G_P_amp = self.G_P * self.X_phi
        G_I_amp = self.G_I_eff * self.X_I
        G_tot = G_P_amp + self.G_E + self.G_D + G_I_amp
        G_tot = max(G_tot, 1e-30)
        return {
            "f_P": G_P_amp / G_tot,
            "f_E": self.G_E / G_tot,
            "f_D": self.G_D / G_tot,
            "f_I": G_I_amp / G_tot,
        }

    def classify_interphase_regime(self) -> str:
        """Classify interphase state based on temperature.

        Returns
        -------
        str
            'frozen': k_BER^int_0 < 1e-6 (exchange effectively frozen)
            'active': k_BER^int_0 >= 1e-6 (exchange active)
        """
        k0_int = self.compute_ber_rate_interphase_equilibrium()
        return "frozen" if k0_int < 1e-6 else "active"

    def get_dual_topological_freezing_temps(self) -> tuple[float, float]:
        """Estimate matrix and interfacial vitrimer topology freezing temperatures.

        T_v is where k_BER_0 * tau_obs ~ 1 (tau_obs ~ 1s).
        T_v = E_a / (R * ln(nu_0))

        Returns
        -------
        tuple of (float, float)
            (T_v^mat, T_v^int) in Kelvin
        """
        T_v_mat = self.E_a / (_R_GAS * max(np.log(max(self.nu_0, 1.0)), 1e-30))
        T_v_int = self.E_a_int / (_R_GAS * max(np.log(max(self.nu_0_int, 1.0)), 1e-30))
        return T_v_mat, T_v_int

    def get_limiting_case(self) -> str:
        """Identify which limiting case is currently active.

        Returns
        -------
        str
            Description of the active limiting case
        """
        fracs = self.get_network_fractions_nc()

        if self.phi < 1e-6:
            return "unfilled vitrimer (phi=0, recovers HVM)"
        elif fracs["f_E"] < 0.01 and fracs["f_D"] < 0.01 and fracs["f_I"] < 0.01:
            return "amplified neo-Hookean (G_P*X only)"
        elif fracs["f_E"] < 0.01 and fracs["f_I"] < 0.01:
            return "filled elastomer (no exchange)"
        elif fracs["f_D"] < 0.01 and fracs["f_I"] < 0.01:
            return "partial vitrimer + amplification"
        elif self.compute_ber_rate_interphase_equilibrium() < 1e-10:
            return "frozen interphase (dead layer)"
        else:
            return "full HVNM (P + E + D + I)"

    def get_relaxation_spectrum(self) -> list[tuple[float, float]]:
        """Return discrete relaxation spectrum [(G_i, tau_i)].

        Returns
        -------
        list of (float, float)
            List of (modulus, relaxation_time) pairs:
            - P-network: (G_P*X, inf) — amplified permanent modulus
            - E-network: (G_E, tau_E_eff)
            - D-network: (G_D, tau_D)
            - I-network: (G_I_eff*X_I, tau_I_eff)
        """
        spectrum = []

        G_P_amp = self.G_P * self.X_phi
        if G_P_amp > 0:
            spectrum.append((G_P_amp, float("inf")))

        if self.G_E > 0:
            tau_E = self.get_vitrimer_relaxation_time()
            spectrum.append((self.G_E, tau_E))

        if self._include_dissociative and self.G_D > 0:
            tau_D = 1.0 / max(self.k_d_D, 1e-30)
            spectrum.append((self.G_D, tau_D))

        G_I_amp = self.G_I_eff * self.X_I
        if G_I_amp > 0:
            tau_I = self.get_interphase_relaxation_time()
            spectrum.append((G_I_amp, tau_I))

        return spectrum

    def arrhenius_plot_data_dual(
        self, T_range: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate dual Arrhenius plot data for both BER rates.

        Parameters
        ----------
        T_range : np.ndarray, optional
            Temperature array (K). Default: 250-450 K, 50 points.

        Returns
        -------
        tuple of (np.ndarray, np.ndarray, np.ndarray)
            (1000/T, log10(k_BER^mat_0), log10(k_BER^int_0))
        """
        if T_range is None:
            T_range = np.linspace(250.0, 450.0, 50)
        T_range = np.asarray(T_range)

        inv_T = 1000.0 / T_range
        log_k_mat = np.array([
            np.log10(float(hvnm_ber_rate_constant_matrix(self.nu_0, self.E_a, Ti)))
            for Ti in T_range
        ])
        log_k_int = np.array([
            np.log10(float(
                hvnm_ber_rate_constant_interphase(self.nu_0_int, self.E_a_int, Ti)
            ))
            for Ti in T_range
        ])

        return inv_T, log_k_mat, log_k_int

    def compute_weissenberg_numbers(self, strain_rate: float) -> tuple[float, float]:
        """Compute Weissenberg numbers for both matrix and interphase.

        Wi^mat = gamma_dot * tau_E_eff
        Wi^int = gamma_dot * tau_I_eff

        Parameters
        ----------
        strain_rate : float
            Applied shear rate (1/s)

        Returns
        -------
        tuple of (float, float)
            (Wi^mat, Wi^int)
        """
        tau_E = self.get_vitrimer_relaxation_time()
        tau_I = self.get_interphase_relaxation_time()
        return strain_rate * tau_E, strain_rate * tau_I

    # =========================================================================
    # String Representation
    # =========================================================================

    def __repr__(self) -> str:
        """Return string representation."""
        parts = [
            f"G_P={self.G_P:.2e}",
            f"G_E={self.G_E:.2e}",
            f"phi={self.phi:.3f}",
        ]
        if self._include_dissociative:
            parts.append(f"G_D={self.G_D:.2e}")
        parts.append(f"kinetics='{self._kinetics}'")
        if self._include_damage:
            parts.append("damage=True")
        if self._include_interfacial_damage:
            parts.append("int_damage=True")
        if self._include_diffusion:
            parts.append("diffusion=True")
        params_str = ", ".join(parts)
        return f"{self.__class__.__name__}({params_str})"
