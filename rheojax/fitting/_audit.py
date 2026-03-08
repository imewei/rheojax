"""Fitting workflow audit summary.

Documents the Protocol→Model mapping, transform registry state,
objective function interfaces, and performance baseline established
during the F1-F10 implementation.

Run directly to get a snapshot:
    python -m rheojax.fitting._audit
"""

from __future__ import annotations


def audit_summary() -> dict:
    """Generate a comprehensive audit of the fitting/transform infrastructure.

    Returns:
        Dictionary with protocol_models, transform_registry, objectives, and notes.
    """
    from rheojax.core.inventory import Protocol, TransformType
    from rheojax.core.registry import Registry

    registry = Registry.get_instance()

    # Protocol → Model mapping
    protocol_models: dict[str, list[str]] = {}
    for proto in Protocol:
        models = registry.find_compatible(protocol=proto)
        protocol_models[proto.value] = models

    # Transform registry state
    transform_state: dict[str, list[str]] = {}
    for ttype in TransformType:
        transforms = registry.find_compatible(transform_type=ttype)
        transform_state[ttype.value] = transforms

    # Objective function interfaces
    objectives = {
        "nlsq_optimize": "rheojax.utils.optimization.nlsq_optimize",
        "create_least_squares_objective": "rheojax.utils.optimization.create_least_squares_objective",
        "nlsq_curve_fit": "rheojax.utils.optimization.nlsq_curve_fit",
    }

    # New Phase 1-4 modules
    new_modules = {
        "fit_result": "rheojax.core.fit_result (FitResult, ModelInfo, ModelComparison)",
        "auto_p0": "rheojax.utils.initialization.auto_p0",
        "physics_checks": "rheojax.utils.physics_checks",
        "model_selection": "rheojax.utils.model_selection",
        "uncertainty": "rheojax.utils.uncertainty",
        "protocol_preprocessing": "rheojax.utils.protocol_preprocessing",
        "prony_conversion": "rheojax.transforms.prony_conversion",
        "cox_merz": "rheojax.transforms.cox_merz",
        "lve_envelope": "rheojax.transforms.lve_envelope",
        "spectrum_inversion": "rheojax.transforms.spectrum_inversion",
    }

    # Phase 4 integration points
    integration = {
        "base_model_fit_kwargs": [
            "auto_init (auto_p0 before fit)",
            "return_result (FitResult instead of self)",
            "check_physics (post-fit physics validation)",
            "uncertainty ('hessian' or 'bootstrap' post-fit CI)",
        ],
        "pipeline_methods": [
            "compare_models(models, criterion) -> Pipeline",
            "get_fit_result() -> FitResult",
        ],
        "writer_extensions": [
            "save_fit_result_hdf5 (rheojax.io.writers.hdf5_writer)",
            "save_fit_result_npz (rheojax.io.writers.npz_writer)",
        ],
    }

    # Performance notes
    performance_notes = {
        "jit_cold_start": "~870ms first fit, <50ms subsequent (model.precompile() available)",
        "model_function_contract": "All 53 models use stateless model_function(X, params, test_mode)",
        "recurrence_pattern": "ODE models use jax.lax.scan or diffrax (no Python loops in hot path)",
        "known_slow": [
            "STZ ODE (8-param, ~3s on N=100)",
            "VLB Nonlocal PDE (51-point, >600s — use scipy method)",
            "EPM SAOS LatticeEPM (NLSQ OOM at L>4)",
        ],
    }

    return {
        "protocol_models": protocol_models,
        "transform_registry": transform_state,
        "objectives": objectives,
        "new_modules": new_modules,
        "integration": integration,
        "performance_notes": performance_notes,
        "total_models": len(registry.get_all_models()),
        "total_transforms": len(registry.get_all_transforms()),
    }


if __name__ == "__main__":
    import json

    result = audit_summary()
    print(json.dumps(result, indent=2, default=str))
