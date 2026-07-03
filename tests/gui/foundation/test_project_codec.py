from pathlib import Path

import numpy as np
import pytest

from rheojax.gui.foundation.project_codec import read_result_arrays, write_result_arrays


def test_flat_scalar_and_array_roundtrip(tmp_path):
    path = tmp_path / "result.hdf5"
    result = {"r_squared": 0.98, "success": True, "message": "ok", "note": None,
              "x_fit": np.array([1.0, 2.0, 3.0])}
    json_shape = write_result_arrays(path, result)
    assert json_shape["r_squared"] == 0.98
    assert json_shape["x_fit"] == {"$hdf5_ref": "x_fit"}
    restored = read_result_arrays(path, json_shape)
    assert restored["r_squared"] == 0.98
    assert restored["success"] is True
    assert restored["note"] is None
    np.testing.assert_array_equal(restored["x_fit"], result["x_fit"])


def test_nested_dict_of_arrays_roundtrip(tmp_path):
    # posterior_samples / sample_stats shape from subprocess_bayesian.py
    path = tmp_path / "nuts_result.hdf5"
    result = {
        "posterior_samples": {"G_p": np.array([1.0, 2.0]), "tau": np.array([0.5, 0.6])},
        "sample_stats": {"energy": np.array([10.0, 11.0])},
        "r_hat": {"G_p": 1.01, "tau": 1.02},
    }
    json_shape = write_result_arrays(path, result)
    assert json_shape["posterior_samples"]["G_p"] == {"$hdf5_ref": "posterior_samples/G_p"}
    assert json_shape["r_hat"] == {"G_p": 1.01, "tau": 1.02}   # no arrays -- stays inline
    restored = read_result_arrays(path, json_shape)
    np.testing.assert_array_equal(restored["posterior_samples"]["G_p"], result["posterior_samples"]["G_p"])
    np.testing.assert_array_equal(restored["sample_stats"]["energy"], result["sample_stats"]["energy"])
    assert restored["r_hat"] == {"G_p": 1.01, "tau": 1.02}


def test_tuple_values_roundtrip_as_tuples(tmp_path):
    # credible_intervals shape from subprocess_bayesian.py: dict[str, tuple[float, float, float]]
    path = tmp_path / "ci_result.hdf5"
    result = {"credible_intervals": {"G_p": (0.9, 1.0, 1.1)}}
    json_shape = write_result_arrays(path, result)
    restored = read_result_arrays(path, json_shape)
    assert restored["credible_intervals"]["G_p"] == (0.9, 1.0, 1.1)
    assert isinstance(restored["credible_intervals"]["G_p"], tuple)


def test_list_of_scalars_stays_a_list(tmp_path):
    path = tmp_path / "list_result.hdf5"
    result = {"iterations": [1, 2, 3]}
    json_shape = write_result_arrays(path, result)
    restored = read_result_arrays(path, json_shape)
    assert restored["iterations"] == [1, 2, 3]
    assert isinstance(restored["iterations"], list)


def test_numpy_scalar_normalized_to_python_scalar(tmp_path):
    path = tmp_path / "npscalar_result.hdf5"
    result = {"chi_squared": np.float64(3.14)}
    json_shape = write_result_arrays(path, result)
    assert isinstance(json_shape["chi_squared"], float)
    assert json_shape["chi_squared"] == pytest.approx(3.14)


def test_unsupported_leaf_type_raises_type_error(tmp_path):
    path = tmp_path / "bad_result.hdf5"

    class Unsupported:
        pass

    with pytest.raises(TypeError):
        write_result_arrays(path, {"bad": Unsupported()})
