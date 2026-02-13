"""Base class for HVM (Hybrid Vitrimer Model).

Provides shared infrastructure for HVM variants:
- Conditional parameter registration (P + E + optional D + optional damage)
- TST kinetics configuration
- Utility methods for vitrimer analysis
- Inherits VLBBase trajectory storage and parameter dict helpers

The HVM models a polymer with three subnetworks:
1. Permanent (P): covalent crosslinks, neo-Hookean elastic (G_P)
2. Exchangeable (E): associative vitrimer bonds with BER kinetics (G_E)
3. Dissociative (D): physical reversible bonds, standard Maxwell (G_D)

References
----------
- Vernerey, Long, & Brighenti (2017). JMPS 107, 1-20.
- Meng, Simon, Niu, McKenna, & Hallinan (2019). Macromolecules 52, 8.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from rheojax.core.jax_config import safe_import_jax
from rheojax.core.parameters import ParameterSet
from rheojax.models.hvm._kernels import (
    hvm_ber_rate_constant,
)
from rheojax.models.vlb._base import VLBBase

jax, jnp = safe_import_jax()

logger = logging.getLogger(__name__)


class HVMBase(VLBBase):
    """Base class for Hybrid Vitrimer Models.

    Extends VLBBase with:
    - 3-subnetwork parameter architecture (P + E + D)
    - TST kinetics for bond exchange reactions
    - Natural-state tensor utilities
    - Vitrimer-specific analysis methods

    Parameters
    ----------
    kinetics : {'stress', 'stretch'}, default 'stress'
        TST coupling mechanism:
        - 'stress': von Mises stress drives BER acceleration
        - 'stretch': chain stretch drives BER acceleration
    include_damage : bool, default False
        Whether to include cooperative damage shielding
    include_dissociative : bool, default True
        Whether to include the dissociative (D) network

    Attributes
    ----------
    kinetics : str
        TST coupling type
    include_damage : bool
        Whether damage is active
    include_dissociative : bool
        Whether D-network is present
    """

    def __init__(
        self,
        kinetics: Literal["stress", "stretch"] = "stress",
        include_damage: bool = False,
        include_dissociative: bool = True,
    ):
        self._kinetics = kinetics
        self._include_damage = include_damage
        self._include_dissociative = include_dissociative

        super().__init__()

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def kinetics(self) -> str:
        """TST coupling type."""
        return self._kinetics

    @property
    def include_damage(self) -> bool:
        """Whether damage is active."""
        return self._include_damage

    @property
    def include_dissociative(self) -> bool:
        """Whether D-network is present."""
        return self._include_dissociative

    @property
    def G_P(self) -> float:
        """Permanent network modulus (Pa)."""
        val = self.parameters.get_value("G_P")
        return float(val) if val is not None else 0.0

    @property
    def G_E(self) -> float:
        """Exchangeable network modulus (Pa)."""
        val = self.parameters.get_value("G_E")
        return float(val) if val is not None else 0.0

    @property
    def nu_0(self) -> float:
        """TST attempt frequency (1/s)."""
        val = self.parameters.get_value("nu_0")
        return float(val) if val is not None else 1e10

    @property
    def E_a(self) -> float:
        """Activation energy for BER (J/mol)."""
        val = self.parameters.get_value("E_a")
        return float(val) if val is not None else 80e3

    @property
    def V_act(self) -> float:
        """Activation volume (m^3/mol)."""
        val = self.parameters.get_value("V_act")
        return float(val) if val is not None else 1e-5

    @property
    def T(self) -> float:
        """Temperature (K)."""
        val = self.parameters.get_value("T")
        return float(val) if val is not None else 300.0

    @property
    def G_D(self) -> float:
        """Dissociative network modulus (Pa)."""
        if not self._include_dissociative:
            return 0.0
        val = self.parameters.get_value("G_D")
        return float(val) if val is not None else 0.0

    @property
    def k_d_D(self) -> float:
        """Dissociative rate (1/s)."""
        if not self._include_dissociative:
            return 1.0  # Avoid division by zero
        val = self.parameters.get_value("k_d_D")
        return float(val) if val is not None else 1.0

    # =========================================================================
    # Parameter Setup
    # =========================================================================

    def _setup_parameters(self):
        """Initialize ParameterSet with conditional parameters.

        Always present (6): G_P, G_E, nu_0, E_a, V_act, T
        When include_dissociative (+2): G_D, k_d_D
        When include_damage (+2): Gamma_0, lambda_crit
        """
        self.parameters = ParameterSet()

        # --- Always present: Permanent + Exchangeable + TST ---
        self.parameters.add(
            name="G_P",
            value=1e4,
            bounds=(0.0, 1e9),
            units="Pa",
            description="Permanent network modulus",
        )
        self.parameters.add(
            name="G_E",
            value=1e4,
            bounds=(0.0, 1e9),
            units="Pa",
            description="Exchangeable network modulus",
        )
        self.parameters.add(
            name="nu_0",
            value=1e10,
            bounds=(1e6, 1e14),
            units="1/s",
            description="TST attempt frequency",
        )
        self.parameters.add(
            name="E_a",
            value=80e3,
            bounds=(20e3, 200e3),
            units="J/mol",
            description="Activation energy for BER",
        )
        self.parameters.add(
            name="V_act",
            value=1e-5,
            bounds=(1e-8, 1e-2),
            units="m^3/mol",
            description="Activation volume",
        )
        self.parameters.add(
            name="T",
            value=300.0,
            bounds=(200.0, 500.0),
            units="K",
            description="Temperature",
        )

        # --- Optional: Dissociative network ---
        if self._include_dissociative:
            self.parameters.add(
                name="G_D",
                value=1e3,
                bounds=(0.0, 1e8),
                units="Pa",
                description="Dissociative network modulus",
            )
            self.parameters.add(
                name="k_d_D",
                value=1.0,
                bounds=(1e-6, 1e6),
                units="1/s",
                description="Dissociative rate",
            )

        # --- Optional: Damage ---
        if self._include_damage:
            self.parameters.add(
                name="Gamma_0",
                value=1e-4,
                bounds=(0.0, 1e-1),
                units="1/s",
                description="Damage rate coefficient",
            )
            self.parameters.add(
                name="lambda_crit",
                value=2.0,
                bounds=(1.001, 10.0),
                units="",
                description="Critical stretch for damage",
            )

    # =========================================================================
    # Vitrimer Utility Methods
    # =========================================================================

    def compute_ber_rate_at_equilibrium(self) -> float:
        """Compute thermal BER rate at zero stress.

        k_BER_0 = nu_0 * exp(-E_a / (R*T))

        Returns
        -------
        float
            Zero-stress BER rate (1/s)
        """
        return float(hvm_ber_rate_constant(self.nu_0, self.E_a, self.T))

    def get_vitrimer_relaxation_time(self) -> float:
        """Effective relaxation time of the exchangeable network.

        tau_E_eff = 1 / (2 * k_BER_0)

        The factor-of-2 arises because both mu^E and mu^E_nat
        relax toward each other at rate k_BER, so their difference
        (which determines stress) decays at 2*k_BER.

        Returns
        -------
        float
            Effective E-network relaxation time (s)
        """
        k0 = self.compute_ber_rate_at_equilibrium()
        return 1.0 / (2.0 * float(jnp.maximum(k0, 1e-30)))

    def get_network_fractions(self) -> dict[str, float]:
        """Compute modulus fractions for each subnetwork.

        Returns
        -------
        dict
            Keys: 'f_P', 'f_E', 'f_D' with values in [0, 1]
        """
        G_tot = self.G_P + self.G_E + self.G_D
        G_tot = float(jnp.maximum(G_tot, 1e-30))
        return {
            "f_P": self.G_P / G_tot,
            "f_E": self.G_E / G_tot,
            "f_D": self.G_D / G_tot,
        }

    def classify_vitrimer_regime(self) -> str:
        """Classify vitrimer state based on temperature.

        The vitrimer topology freeze transition temperature T_v is
        where k_BER_0 * tau_observation ~ 1. Below T_v, exchange
        is frozen (glassy). Above T_v, exchange is active.

        Returns
        -------
        str
            'glassy': k_BER_0 < 1e-6 (exchange effectively frozen)
            'rubbery': 1e-6 <= k_BER_0 < 1e2 (active exchange)
            'flow': k_BER_0 >= 1e2 (liquid-like exchange)
        """
        k0 = self.compute_ber_rate_at_equilibrium()
        if k0 < 1e-6:
            return "glassy"
        elif k0 < 1e2:
            return "rubbery"
        else:
            return "flow"

    def arrhenius_plot_data(
        self, T_range: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate Arrhenius plot data for BER rate.

        Parameters
        ----------
        T_range : np.ndarray, optional
            Temperature array (K). Default: 250-450 K, 50 points.

        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            (1000/T, log10(k_BER_0)) for plotting
        """
        if T_range is None:
            T_range = np.linspace(250.0, 450.0, 50)
        T_range = np.asarray(T_range)

        inv_T = 1000.0 / T_range
        log_k = np.array([
            np.log10(float(hvm_ber_rate_constant(self.nu_0, self.E_a, T_i)))
            for T_i in T_range
        ])

        return inv_T, log_k

    def get_limiting_case(self) -> str:
        """Identify which limiting case is currently active.

        Returns
        -------
        str
            Description of the active limiting case
        """
        fracs = self.get_network_fractions()

        if fracs["f_E"] < 0.01 and fracs["f_D"] < 0.01:
            return "neo-Hookean (G_P only)"
        elif fracs["f_P"] < 0.01 and fracs["f_E"] < 0.01:
            return "Maxwell (G_D only)"
        elif fracs["f_E"] < 0.01:
            return "Zener/SLS (G_P + G_D)"
        elif fracs["f_D"] < 0.01 and fracs["f_P"] < 0.01:
            return "pure vitrimer (G_E only)"
        elif fracs["f_D"] < 0.01:
            return "partial vitrimer (G_P + G_E, Meng 2019)"
        else:
            return "full HVM (G_P + G_E + G_D)"

    def get_relaxation_spectrum(self) -> list[tuple[float, float]]:
        """Return discrete relaxation spectrum [(G_i, tau_i)].

        Returns
        -------
        list of (float, float)
            List of (modulus, relaxation_time) pairs.
            - P-network: (G_P, inf)
            - E-network: (G_E, tau_E_eff)
            - D-network: (G_D, tau_D)
        """
        spectrum = []

        if self.G_P > 0:
            spectrum.append((self.G_P, float("inf")))

        if self.G_E > 0:
            tau_E = self.get_vitrimer_relaxation_time()
            spectrum.append((self.G_E, tau_E))

        if self._include_dissociative and self.G_D > 0:
            tau_D = 1.0 / max(self.k_d_D, 1e-30)
            spectrum.append((self.G_D, tau_D))

        return spectrum

    # =========================================================================
    # String Representation
    # =========================================================================

    def __repr__(self) -> str:
        """Return string representation."""
        parts = [f"G_P={self.G_P:.2e}", f"G_E={self.G_E:.2e}"]
        if self._include_dissociative:
            parts.append(f"G_D={self.G_D:.2e}")
        parts.append(f"kinetics='{self._kinetics}'")
        if self._include_damage:
            parts.append("damage=True")
        params_str = ", ".join(parts)
        return f"{self.__class__.__name__}({params_str})"
