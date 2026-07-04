from rheojax.gui.foundation.invalidation import invalidate_downstream
from rheojax.gui.foundation.state import (
    ActiveJobsState,
    AppState,
    FitState,
    JobHistoryState,
    JobResultRef,
    NlsqConfig,
    NutsConfig,
    ParameterConfig,
    PipelineState,
    PipelineStepConfig,
    UiState,
)


def test_nlsq_config_defaults():
    cfg = NlsqConfig()
    assert cfg.multi_start is False
    assert cfg.n_starts == 8
    assert cfg.parameters == []


def test_nuts_config_defaults():
    cfg = NutsConfig()
    assert cfg.run_nuts is True
    assert cfg.num_warmup == 500
    assert cfg.num_samples == 1000
    assert cfg.num_chains == 4
    assert cfg.target_accept == 0.8
    assert cfg.seed == 0
    assert cfg.warm_start is True
    assert cfg.priors == {}


def test_parameter_config_fields():
    p = ParameterConfig(name="G0", value=1.0, lower=0.0, upper=10.0, fixed=False)
    assert p.name == "G0"
    assert p.fixed is False


def test_fit_state_uses_typed_nlsq_nuts_config():
    fit = FitState(
        protocol="oscillation",
        model_key="maxwell",
        model_config={},
        data_ref="d1",
        column_map={},
        control_vars={},
        nlsq_config=NlsqConfig(n_starts=3),
        nuts_config=NutsConfig(run_nuts=False),
        nlsq_result=None,
        nuts_result=None,
        step=0,
        revision=0,
    )
    assert fit.nlsq_config.n_starts == 3
    assert fit.nuts_config.run_nuts is False


def test_changing_model_clears_downstream():
    fit = FitState(
        protocol="oscillation",
        model_key="maxwell",
        model_config={"n_modes": 3},
        data_ref="d1",
        column_map={"x": 0},
        control_vars={"gamma_dot": 1.0},
        nlsq_config=NlsqConfig(),
        nlsq_result=object(),
        nuts_result={"rhat": 1.0},
        step=4,
        revision=0,
    )
    new = invalidate_downstream(fit, changed="model_key")
    assert new.nlsq_result is None and new.nuts_result is None
    assert new.data_ref is None and new.column_map == {}
    # control_vars and model_config must also be cleared on model change
    assert new.control_vars == {}
    assert new.model_config == {}
    assert new.revision == 1


def test_changing_protocol_clears_control_vars_and_model_config():
    fit = FitState(
        protocol="oscillation",
        model_key="maxwell",
        model_config={"n_modes": 3},
        data_ref="d2",
        column_map={"x": 0},
        control_vars={"omega": 1.0},
        nlsq_config=NlsqConfig(),
        nlsq_result=object(),
        nuts_result=None,
        step=2,
        revision=0,
    )
    new = invalidate_downstream(fit, changed="protocol")
    assert new.control_vars == {}
    assert new.model_config == {}
    assert new.nlsq_result is None
    assert new.revision == 1


def test_changing_columnmap_keeps_data_clears_fits():
    fit = FitState(
        protocol="oscillation",
        model_key="maxwell",
        model_config={},
        data_ref="d1",
        column_map={"x": 0},
        control_vars={},
        nlsq_config=NlsqConfig(),
        nlsq_result=object(),
        nuts_result=None,
        step=3,
        revision=2,
    )
    new = invalidate_downstream(fit, changed="column_map")
    assert new.data_ref == "d1"  # data kept
    assert new.nlsq_result is None  # fit cleared
    assert new.revision == 3


def test_pipeline_step_config_fields():
    step = PipelineStepConfig(id="s1", step_type="fit", config={"run_nuts": True})
    assert step.step_type == "fit"


def test_job_result_ref():
    ref = JobResultRef(result_id="abc123")
    assert ref.result_id == "abc123"


def test_pipeline_state_defaults():
    ps = PipelineState()
    assert ps.steps == []
    assert ps.selected_dataset_ids == []
    assert ps.job_id is None


def test_ui_state_defaults():
    ui = UiState()
    assert ui.mode == "fit"
    assert ui.theme == "system"
    assert ui.inspector_tab == "log"


def test_app_state_has_pipeline_and_job_slices():
    state = AppState()
    assert isinstance(state.pipeline, PipelineState)
    assert isinstance(state.active_jobs, ActiveJobsState)
    assert isinstance(state.job_history, JobHistoryState)
    assert isinstance(state.ui, UiState)
    assert state.ui.mode == "fit"
