#!/usr/bin/env python3
"""Quick reference guide for plotting TRIOS complex modulus data without warnings.

This script demonstrates the correct approaches to avoid matplotlib warnings when
plotting complex modulus data loaded from TRIOS files.

The Problem:
-----------
TRIOS frequency sweep data contains both G' (Storage) and G'' (Loss) moduli.
RheoJAX loads this as complex modulus: G* = G' + i·G''

Matplotlib CANNOT plot complex-valued arrays directly, which causes:
- ComplexWarning: Casting complex values to real discards the imaginary part
- UserWarning: Data has no positive values, and therefore cannot be log-scaled

Solutions:
---------
Use one of the three approaches demonstrated below.
"""

import matplotlib.pyplot as plt
import numpy as np

from rheojax.io import load_trios
from rheojax.visualization import plot_rheo_data

# Load TRIOS data (returns complex modulus)
data_list = load_trios("../data/experimental/frequency_sweep_tts.txt")
data = data_list[0]  # First temperature segment

print("=" * 70)
print("TRIOS Complex Modulus Plotting - Quick Reference")
print("=" * 70)
print(f"\nLoaded {len(data_list)} temperature segments")
print(f"Data type: {data.y.dtype}")
print(f"Is complex: {np.iscomplexobj(data.y)}")
print()


# ============================================================================
# ❌ WRONG: Direct plotting (DO NOT DO THIS)
# ============================================================================
print("❌ WRONG APPROACH (causes warnings):")
print("-" * 70)
print("plt.loglog(data.x, data.y)  # <- Will trigger ComplexWarning")
print("plt.tight_layout()           # <- Warnings appear here")
print()


# ============================================================================
# ✅ SOLUTION 1: Use RheoJAX visualization (RECOMMENDED)
# ============================================================================
print("✅ SOLUTION 1: Use RheoJAX visualization (RECOMMENDED)")
print("-" * 70)
print("from rheojax.visualization import plot_rheo_data")
print("fig, axes = plot_rheo_data(data)")
print("plt.show()")
print()

fig1, axes = plot_rheo_data(data)
plt.suptitle(
    f"Solution 1: RheoJAX visualization\nTemp: {data.metadata.get('temperature', 'N/A')}°C",
    y=1.02,
)
plt.tight_layout()
plt.savefig("solution1_rheojax_plot.png", dpi=150, bbox_inches="tight")
print("✓ Figure saved: solution1_rheojax_plot.png")
plt.close()


# ============================================================================
# ✅ SOLUTION 2: Manual component extraction with np.real/np.imag
# ============================================================================
print("\n✅ SOLUTION 2: Manual extraction with np.real() and np.imag()")
print("-" * 70)
print("omega = data.x")
print("G_prime = np.real(data.y)        # Storage modulus (G')")
print("G_double_prime = np.imag(data.y) # Loss modulus (G'')")
print('plt.loglog(omega, G_prime, label="G\'")')
print("plt.loglog(omega, G_double_prime, label='G\"')")
print()

fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

omega = data.x
G_prime = np.real(data.y)
G_double_prime = np.imag(data.y)

ax1.loglog(omega, G_prime, "o-", label="G' (Storage)", color="C0")
ax1.set_xlabel("Frequency (rad/s)")
ax1.set_ylabel("G' (Pa)")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.loglog(omega, G_double_prime, "s-", label='G" (Loss)', color="C1")
ax2.set_xlabel("Frequency (rad/s)")
ax2.set_ylabel('G" (Pa)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle(
    f"Solution 2: Manual extraction\nTemp: {data.metadata.get('temperature', 'N/A')}°C"
)
plt.tight_layout()
plt.savefig("solution2_manual_extraction.png", dpi=150, bbox_inches="tight")
print("✓ Figure saved: solution2_manual_extraction.png")
plt.close()


# ============================================================================
# ✅ SOLUTION 3: Use RheoData convenience properties (NEW in v0.2.0)
# ============================================================================
print("\n✅ SOLUTION 3: Use RheoData convenience properties (NEW in v0.2.0)")
print("-" * 70)
print("omega = data.x")
print("G_prime = data.y_real              # or data.storage_modulus")
print("G_double_prime = data.y_imag       # or data.loss_modulus")
print("tan_delta = data.tan_delta         # G''/G' ratio")
print('plt.loglog(omega, G_prime, label="G\'")')
print()

fig3, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))

# Using convenience properties
omega = data.x
G_prime = data.y_real  # Equivalent to data.storage_modulus
G_double_prime = data.y_imag  # Equivalent to data.loss_modulus
tan_delta = data.tan_delta

ax1.loglog(omega, G_prime, "o-", label="G' (Storage)", color="C0")
ax1.set_ylabel("G' (Pa)")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.loglog(omega, G_double_prime, "s-", label='G" (Loss)', color="C1")
ax2.set_ylabel('G" (Pa)')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3.semilogx(omega, tan_delta, "d-", label="tan δ = G\"/G'", color="C2")
ax3.set_xlabel("Frequency (rad/s)")
ax3.set_ylabel("tan δ (dimensionless)")
ax3.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="tan δ = 1")
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.suptitle(
    f"Solution 3: RheoData properties\nTemp: {data.metadata.get('temperature', 'N/A')}°C"
)
plt.tight_layout()
plt.savefig("solution3_rhedata_properties.png", dpi=150, bbox_inches="tight")
print("✓ Figure saved: solution3_rhedata_properties.png")
plt.close()


# ============================================================================
# Data Quality Report
# ============================================================================
print("\n" + "=" * 70)
print("DATA QUALITY REPORT")
print("=" * 70)

all_G_prime = np.concatenate([d.y_real for d in data_list])
all_G_double_prime = np.concatenate([d.y_imag for d in data_list])

print("\nG' (Storage Modulus):")
print(f"  Range: {all_G_prime.min():.1f} - {all_G_prime.max():.1f} Pa")
print(f"  All positive: {np.all(all_G_prime > 0)}")

print("\nG'' (Loss Modulus):")
print(f"  Range: {all_G_double_prime.min():.1f} - {all_G_double_prime.max():.1f} Pa")
print(f"  All positive: {np.all(all_G_double_prime > 0)}")

tan_delta_all = all_G_double_prime / all_G_prime
print("\nLoss Tangent (tan δ = G''/G'):")
print(f"  Range: {tan_delta_all.min():.3f} - {tan_delta_all.max():.3f}")
print(f"  Mean: {tan_delta_all.mean():.3f}")
print(
    f"  Material type: {'Elastic-dominant (solid-like)' if tan_delta_all.mean() < 1 else 'Viscous-dominant (liquid-like)'}"
)

if np.all(all_G_prime > 0) and np.all(all_G_double_prime > 0):
    print("\n✓ All data values are positive - safe for log-scale plotting!")
else:
    print("\n⚠️  Warning: Some values are non-positive")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("\nAvailable RheoData properties for complex modulus:")
print("  • data.y_real           → G' (Storage modulus)")
print("  • data.y_imag           → G'' (Loss modulus)")
print("  • data.storage_modulus  → Alias for y_real")
print("  • data.loss_modulus     → Alias for y_imag")
print("  • data.tan_delta        → G''/G' ratio")
print("  • data.modulus          → |G*| magnitude")
print("  • data.phase            → Phase angle")
print()
print("To fix warnings in your notebook, replace:")
print("  plt.loglog(data.x, data.y)  # ❌ WRONG")
print("with one of:")
print("  plot_rheo_data(data)              # ✅ Easiest")
print("  plt.loglog(data.x, data.y_real)   # ✅ Manual G'")
print("  plt.loglog(data.x, np.real(data.y))  # ✅ Alternative")
print("=" * 70)
