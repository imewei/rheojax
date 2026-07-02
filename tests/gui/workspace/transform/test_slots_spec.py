from rheojax.gui.workspace.transform.slots_spec import transform_slots


def test_cox_merz_two_typed_slots():
    s = transform_slots("cox_merz")
    assert [
        (x.name, x.accepts, x.is_list) for x in s
    ] == [
        ("oscillation", "oscillation", False),
        ("flow_curve", "flow_curve", False),
    ]


def test_mastercurve_is_list():
    s = transform_slots("mastercurve")
    assert len(s) == 1 and s[0].is_list is True


def test_default_single_slot():
    s = transform_slots("fft_analysis")
    assert len(s) == 1 and s[0].is_list is False
