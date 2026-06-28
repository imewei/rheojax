import pathlib

def test_gui_clean():
    bad = [str(p) for p in pathlib.Path("rheojax/gui").rglob("*.py")
           if any(s in p.read_text() for s in
                  ("deformation_mode", "poisson_ratio", "get_supported_deformation_modes"))]
    assert bad == [], bad
