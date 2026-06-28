# tests/io/test_dmta_rejection.py
import pandas as pd, pytest
from rheojax.io._exceptions import UnsupportedDataError
from rheojax.io.readers.auto import auto_load

def test_tensile_columns_rejected(tmp_path):
    f = tmp_path / "dma.csv"
    pd.DataFrame({"frequency": [1.0,2,3],          # 'frequency' (auto-detected), not 'freq'
                  "Tensile Storage Modulus": [1e6,2e6,3e6],
                  "Tensile Loss Modulus": [1e5,2e5,3e5]}).to_csv(f, index=False)
    with pytest.raises(UnsupportedDataError):
        auto_load(str(f))
