#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/b80985/Projects/Rheo')

import jax.numpy as jnp
import numpy as np

# Test fractional_maxwell_model
from rheo.models.fractional_maxwell_model import FractionalMaxwellModel

print("Testing FractionalMaxwellModel...")
model = FractionalMaxwellModel()

# Set parameters
model.parameters.set_value('c1', 1e5)
model.parameters.set_value('alpha', 0.5)
model.parameters.set_value('beta', 0.7)
model.parameters.set_value('tau', 1.0)

# Test relaxation
t = jnp.logspace(-2, 2, 10)
try:
    result = model._predict_relaxation_jax(t, 1e5, 0.5, 0.7, 1.0)
    print(f"✓ Relaxation test passed! Result shape: {result.shape}")
except Exception as e:
    print(f"✗ Relaxation test failed: {e}")
    sys.exit(1)

# Test creep
try:
    result = model._predict_creep_jax(t, 1e5, 0.5, 0.7, 1.0)
    print(f"✓ Creep test passed! Result shape: {result.shape}")
except Exception as e:
    print(f"✗ Creep test failed: {e}")
    sys.exit(1)

# Test oscillation
omega = jnp.logspace(-2, 2, 10)
try:
    result = model._predict_oscillation_jax(omega, 1e5, 0.5, 0.7, 1.0)
    print(f"✓ Oscillation test passed! Result shape: {result.shape}")
except Exception as e:
    print(f"✗ Oscillation test failed: {e}")
    sys.exit(1)

print("\nAll tests passed! ✓")
