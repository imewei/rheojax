from rheojax.gui.workspace.pipeline.models import (
    DatasetArtifact,
    FileArtifact,
    FitArtifact,
    FitStepResult,
    PhaseResult,
    PipelineRunResult,
)


def test_phase_result_defaults():
    p = PhaseResult(status="pending")
    assert p.result is None
    assert p.error is None


def test_fit_step_result_nlsq_only():
    fsr = FitStepResult(
        nlsq=PhaseResult(status="completed", result={"r_squared": 0.9}), nuts=None
    )
    assert fsr.nuts is None
    assert fsr.nlsq.status == "completed"


def test_fit_step_result_nlsq_success_nuts_failure_independent():
    fsr = FitStepResult(
        nlsq=PhaseResult(status="completed", result={"r_squared": 0.9}),
        nuts=PhaseResult(status="failed", error="NUTS divergence"),
    )
    assert fsr.nlsq.status == "completed"
    assert fsr.nuts.status == "failed"
    assert fsr.nuts.error == "NUTS divergence"


def test_pipeline_run_result_error_field():
    r = PipelineRunResult(step_results={}, status="failed", error="step s1 raised")
    assert r.error == "step s1 raised"


def test_artifact_dataclasses():
    da = DatasetArtifact(dataset_id="d1", source_step_id="s1", produced_at_revision=0)
    fa = FitArtifact(
        step_id="s2",
        source_dataset_id="d1",
        result=FitStepResult(nlsq=PhaseResult(status="completed"), nuts=None),
    )
    fla = FileArtifact(step_id="s3", paths=["/tmp/out.csv"])
    assert da.dataset_id == "d1"
    assert fa.result.nlsq.status == "completed"
    assert fla.paths == ["/tmp/out.csv"]
