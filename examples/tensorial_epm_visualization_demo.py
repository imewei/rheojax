"""Demonstration of tensorial EPM visualization capabilities.

This script demonstrates all the visualization functions for the TensorialEPM model,
including:
- Auto-detection of scalar vs tensorial stress fields
- 3-panel tensorial field visualization
- Normal stress difference plots
- Von Mises effective stress visualization
- Normal stress ratio analysis
- Animated stress evolution

Run with: uv run python examples/tensorial_epm_visualization_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt

from rheojax.core.jax_config import safe_import_jax
from rheojax.models.epm.tensor import TensorialEPM
from rheojax.core.data import RheoData
from rheojax.visualization.epm_plots import (
    plot_lattice_fields,
    plot_tensorial_fields,
    plot_normal_stress_field,
    plot_von_mises_field,
    plot_normal_stress_ratio,
    animate_tensorial_evolution,
)

jax, jnp = safe_import_jax()


def demo_auto_detection():
    """Demonstrate auto-detection of scalar vs tensorial stress."""
    print("=== Demo 1: Auto-Detection of Scalar vs Tensorial ===\n")

    L = 32

    # Scalar stress (legacy scalar EPM)
    stress_scalar = np.random.randn(L, L)
    thresholds = np.abs(np.random.randn(L, L)) + 0.5

    print("Plotting scalar stress field...")
    fig1 = plot_lattice_fields(stress_scalar, thresholds)
    fig1.savefig("/tmp/epm_scalar_auto.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print("Saved to /tmp/epm_scalar_auto.png\n")

    # Tensorial stress
    stress_tensor = np.random.randn(3, L, L)

    print("Plotting tensorial stress field...")
    fig2 = plot_lattice_fields(stress_tensor, thresholds)
    fig2.savefig("/tmp/epm_tensor_auto.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("Saved to /tmp/epm_tensor_auto.png\n")


def demo_tensorial_fields():
    """Demonstrate 3-panel tensorial field visualization."""
    print("=== Demo 2: Tensorial Field Components ===\n")

    L = 32

    # Simulate a shear-dominant stress state
    stress = np.zeros((3, L, L))
    x, y = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')

    # σ_xx: Extensional pattern
    stress[0] = 0.5 * np.sin(2 * np.pi * x / L) * np.cos(2 * np.pi * y / L)

    # σ_yy: Compressional pattern (opposite sign)
    stress[1] = -0.5 * np.sin(2 * np.pi * x / L) * np.cos(2 * np.pi * y / L)

    # σ_xy: Shear pattern (dominant)
    stress[2] = 2.0 * np.cos(2 * np.pi * x / L) * np.sin(2 * np.pi * y / L)

    print("Plotting tensorial field components...")
    fig, axes = plot_tensorial_fields(stress)
    fig.savefig("/tmp/epm_tensorial_fields.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved to /tmp/epm_tensorial_fields.png\n")


def demo_normal_stress():
    """Demonstrate normal stress difference visualization."""
    print("=== Demo 3: Normal Stress Difference N₁ ===\n")

    L = 32
    stress = np.zeros((3, L, L))
    x, y = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')

    # Create significant N₁ = σ_xx - σ_yy
    stress[0] = 1.0 * np.sin(2 * np.pi * x / L)  # σ_xx
    stress[1] = -0.5 * np.sin(2 * np.pi * x / L)  # σ_yy
    stress[2] = 0.3 * np.random.randn(L, L)  # σ_xy (noise)

    print("Plotting N₁ field...")
    fig, ax = plot_normal_stress_field(stress, nu=0.48)
    fig.savefig("/tmp/epm_normal_stress.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved to /tmp/epm_normal_stress.png\n")


def demo_von_mises():
    """Demonstrate von Mises effective stress visualization."""
    print("=== Demo 4: Von Mises Effective Stress ===\n")

    L = 32
    stress = np.zeros((3, L, L))
    x, y = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')

    # Create localized yielding region
    r_sq = (x - L/2)**2 + (y - L/2)**2
    stress[0] = 0.5 * np.exp(-r_sq / (L/4)**2)
    stress[1] = -0.5 * np.exp(-r_sq / (L/4)**2)
    stress[2] = 1.5 * np.exp(-r_sq / (L/3)**2)

    # Yield thresholds with disorder
    thresholds = np.ones((L, L)) + 0.2 * np.random.randn(L, L)

    print("Plotting von Mises stress and normalized yield map...")
    fig, axes = plot_von_mises_field(stress, thresholds, nu=0.48)
    fig.savefig("/tmp/epm_von_mises.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved to /tmp/epm_von_mises.png\n")


def demo_normal_stress_ratio():
    """Demonstrate normal stress ratio vs shear rate."""
    print("=== Demo 5: Normal Stress Ratio Analysis ===\n")

    # Simulate flow curve data with power-law N₁
    shear_rates = np.logspace(-2, 2, 20)

    # Power-law N₁ ~ γ̇^1.5
    N1 = 0.5 * shear_rates**1.5 + 0.1 * np.random.randn(20)

    # Shear stress ~ γ̇^0.8 (shear thinning)
    sigma_xy = shear_rates**0.8 + 0.05 * np.random.randn(20)

    print("Plotting N₁/σ_xy vs shear rate...")
    fig, ax = plot_normal_stress_ratio(shear_rates, N1, sigma_xy)
    fig.savefig("/tmp/epm_normal_stress_ratio.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved to /tmp/epm_normal_stress_ratio.png\n")


def demo_animation():
    """Demonstrate animated stress evolution."""
    print("=== Demo 6: Animated Stress Evolution ===\n")

    L = 16
    T = 20

    # Create synthetic stress evolution (shear startup)
    time = np.linspace(0, 2, T)
    stress_history = np.zeros((T, 3, L, L))

    x, y = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')

    for t_idx in range(T):
        # Build up stress over time
        amplitude = 1.0 - np.exp(-time[t_idx])

        stress_history[t_idx, 0] = 0.3 * amplitude * np.sin(2 * np.pi * x / L)
        stress_history[t_idx, 1] = -0.3 * amplitude * np.sin(2 * np.pi * x / L)
        stress_history[t_idx, 2] = amplitude * np.cos(2 * np.pi * x / L) * np.sin(2 * np.pi * y / L)

    history = {
        'stress': stress_history,
        'time': time
    }

    print("Creating animation (all components)...")
    anim = animate_tensorial_evolution(history, component='all', interval=100)
    anim.save("/tmp/epm_evolution_all.gif", writer='pillow', fps=10)
    plt.close(anim._fig)
    print("Saved to /tmp/epm_evolution_all.gif\n")

    print("Creating animation (von Mises)...")
    anim_vm = animate_tensorial_evolution(history, component='vm', interval=100, nu=0.48)
    anim_vm.save("/tmp/epm_evolution_vm.gif", writer='pillow', fps=10)
    plt.close(anim_vm._fig)
    print("Saved to /tmp/epm_evolution_vm.gif\n")


def main():
    """Run all visualization demos."""
    print("\n" + "="*60)
    print("Tensorial EPM Visualization Demonstration")
    print("="*60 + "\n")

    demo_auto_detection()
    demo_tensorial_fields()
    demo_normal_stress()
    demo_von_mises()
    demo_normal_stress_ratio()
    demo_animation()

    print("="*60)
    print("All demonstrations complete!")
    print("Output files saved to /tmp/epm_*.png and /tmp/epm_*.gif")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
