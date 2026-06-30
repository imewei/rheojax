from rheojax.gui.foundation.invalidation import invalidate_downstream
from rheojax.gui.foundation.state import FitState


def test_changing_model_clears_downstream():
    fit = FitState(protocol="oscillation", model_key="maxwell", model_config={},
                   data_ref="d1", column_map={"x": 0}, control_vars={}, nlsq_config={},
                   nlsq_result=object(), nuts_result={"rhat": 1.0}, step=4, revision=0)
    new = invalidate_downstream(fit, changed="model_key")
    assert new.nlsq_result is None and new.nuts_result is None
    assert new.data_ref is None and new.column_map == {}
    assert new.revision == 1

def test_changing_columnmap_keeps_data_clears_fits():
    fit = FitState(protocol="oscillation", model_key="maxwell", model_config={},
                   data_ref="d1", column_map={"x": 0}, control_vars={}, nlsq_config={},
                   nlsq_result=object(), nuts_result=None, step=3, revision=2)
    new = invalidate_downstream(fit, changed="column_map")
    assert new.data_ref == "d1"          # data kept
    assert new.nlsq_result is None       # fit cleared
    assert new.revision == 3
