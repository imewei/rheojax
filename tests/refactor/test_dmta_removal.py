# tests/refactor/test_dmta_removal.py
import pathlib
RJ = pathlib.Path("rheojax")
IDENTS = ("DeformationMode", "deformation_mode", "poisson_ratio",
          "modulus_conversion", "convert_modulus", "convert_rheodata",
          "POISSON_PRESETS", "is_tensile", 'modulus_type')

def _hits(subdir, idents):
    return sorted(str(p) for p in (RJ / subdir).rglob("*.py")
                  if any(i in p.read_text() for i in idents))

def test_no_model_imports_deformationmode():
    assert _hits("models", ("DeformationMode",)) == []
