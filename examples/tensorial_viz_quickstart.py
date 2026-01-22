"""Quick start guide for tensorial EPM visualization.

This example demonstrates the simplest use cases for each visualization function.
"""

import numpy as np
from rheojax.visualization import (
    plot_lattice_fields,
    plot_tensorial_fields,
    plot_normal_stress_field,
    plot_von_mises_field,
    plot_normal_stress_ratio,
    animate_tensorial_evolution,
)

# Example 1: Auto-detection (works with both scalar and tensorial stress)
L = 32
stress_tensor = np.random.randn(3, L, L)  # Tensorial: (3, L, L)
thresholds = np.abs(np.random.randn(L, L)) + 0.5

fig1 = plot_lattice_fields(stress_tensor, thresholds)
fig1.savefig("example1_auto_detection.png")

# Example 2: Visualize all stress components
fig2, axes = plot_tensorial_fields(stress_tensor)
fig2.savefig("example2_components.png")

# Example 3: Visualize normal stress difference N₁
fig3, ax = plot_normal_stress_field(stress_tensor, nu=0.48)
fig3.savefig("example3_N1.png")

# Example 4: Visualize von Mises effective stress
fig4, axes = plot_von_mises_field(stress_tensor, thresholds, nu=0.48)
fig4.savefig("example4_von_mises.png")

# Example 5: Plot normal stress ratio from flow curve
shear_rates = np.logspace(-2, 2, 20)
N1 = 0.5 * shear_rates**1.5
sigma_xy = shear_rates**0.8

fig5, ax = plot_normal_stress_ratio(shear_rates, N1, sigma_xy)
fig5.savefig("example5_N1_ratio.png")

# Example 6: Animate stress evolution
T = 20
stress_history = np.random.randn(T, 3, L, L)
time = np.linspace(0, 2, T)

history = {'stress': stress_history, 'time': time}

# All components
anim1 = animate_tensorial_evolution(history, component='all')
anim1.save("example6_evolution.gif", writer='pillow')

# Von Mises only
anim2 = animate_tensorial_evolution(history, component='vm', nu=0.48)
anim2.save("example6_von_mises_evolution.gif", writer='pillow')

print("All examples saved successfully!")
