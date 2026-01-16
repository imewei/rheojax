# Protocol-Driven Inventory System (Models & Transforms)

## Problem Statement
Currently, RheoJAX determines capabilities through heuristic checks (presence of methods) and implicit assumptions. This makes it difficult for users to:
1. Discover which **models** support a specific experimental protocol (e.g., "Find all models that support LAOS").
2. Discover which **transforms** are applicable to their data type (e.g., "What transforms work on time-domain LAOS data?").
3. Ensure type safety and correctness when building analysis pipelines.

## Proposed Solution
We propose a **Protocol-Driven Inventory System** that explicitly classifies both **Models** and **Transforms**. This formalizes the relationship between components and their supported operations.

### 1. Dual Classification System

The inventory will be divided into two primary classes:

1.  **Models**: Constitutive equations that predict material response (Fit & Predict).
    - Categorized by **Protocol** (Flow, Creep, Relaxation, etc.)
2.  **Transforms**: Data processing operations (Transform & Inverse).
    - Categorized by **TransformType** (Spectral, Superposition, Decomposition, etc.)

### 2. Core Definitions

#### A. Protocols (For Models)
Formalize `TestMode` as a rich `Protocol` definition.

```python
class Protocol(str, Enum):
    FLOW_CURVE = "flow_curve"       # Steady shear viscosity vs shear rate
    CREEP = "creep"                 # Strain vs time at constant stress
    RELAXATION = "relaxation"       # Stress vs time at constant strain
    STARTUP = "startup"             # Stress growth vs time at constant rate
    SAOS = "saos"                   # Small Amplitude Oscillatory Shear (G', G'')
    LAOS = "laos"                   # Large Amplitude Oscillatory Shear (Lissajous)
```

#### B. Transform Types (For Transforms)
Categorize the 7 data transforms based on their mathematical operation.

```python
class TransformType(str, Enum):
    SPECTRAL = "spectral"           # Time <-> Frequency domain (FFT)
    SUPERPOSITION = "superposition" # Shift data to master curve (TTS, SRFS)
    DECOMPOSITION = "decomposition" # Split signal into components (SPP)
    ANALYSIS = "analysis"           # Extract metrics (Mutation Number, OWChirp)
    PROCESSING = "processing"       # Data cleaning/smoothing (Smooth Derivative)
```

### 3. Enhanced Registration

Both `ModelRegistry` and `TransformRegistry` will accept capability metadata.

#### Model Registration
```python
@ModelRegistry.register(
    name="sgr_conventional",
    protocols=[Protocol.FLOW_CURVE, Protocol.LAOS, Protocol.STARTUP],
    tags=["physics-based", "thixotropy", "yield-stress"]
)
class SGRConventional(BaseModel): ...
```

#### Transform Registration
```python
@TransformRegistry.register(
    name="mastercurve",
    type=TransformType.SUPERPOSITION,
    input_domain="frequency",  # or 'time', 'any'
    tags=["tts", "wlf", "arrhenius"]
)
class Mastercurve(BaseTransform): ...
```

### 4. Inventory Query API

The unified `Registry` will expose query methods for both classes.

```python
# Models: "I have LAOS data, what models can I use?"
laos_models = ModelRegistry.find(protocol=Protocol.LAOS)
# -> ['sgr_conventional', 'stz_conventional', 'spp_yield_stress']

# Transforms: "I need to perform superposition (TTS/SRFS)"
superposition_transforms = TransformRegistry.find(type=TransformType.SUPERPOSITION)
# -> ['mastercurve', 'srfs']

# Global Inventory
inventory = Registry.inventory()
# -> {
#   'models': {'maxwell': {'protocols': ['relaxation', ...]}, ...},
#   'transforms': {'fft': {'type': 'spectral'}, ...}
# }
```

### 5. Implementation Plan

#### Phase 1: Core Definitions
- Define `Protocol` and `TransformType` enums in `rheojax.core.inventory`.
- Update `PluginInfo` dataclass to support generic metadata schemas.

#### Phase 2: Registry Update
- Update `Registry.register` to accept `protocols` (for models) and `transform_type` (for transforms).
- Implement validation logic for both classes.

#### Phase 3: Migration
- **Models**: Update 25 models with protocol lists.
- **Transforms**: Update 7 transforms with types:
    - FFT -> `SPECTRAL`
    - Mastercurve -> `SUPERPOSITION`
    - SRFS -> `SUPERPOSITION`
    - SPP -> `DECOMPOSITION`
    - Mutation Number -> `ANALYSIS`
    - OWChirp -> `ANALYSIS`
    - Smooth Derivative -> `PROCESSING`

#### Phase 4: User-Facing API
- Implement `find()` methods.
- Add CLI command `rheojax inventory` to show the full matrix.

## Benefits
- **Unified Discovery**: A consistent way to find any tool in the package.
- **Clarity**: Explicitly distinguishing between "fitting a model" and "transforming data".
- **Future-Proofing**: Easy to add new protocols (e.g., "Dielectric") or transform types.
