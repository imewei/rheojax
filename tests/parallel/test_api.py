"""Tests for parallel public API."""

from unittest.mock import patch

import pytest


# Module-level function for pickling
def _test_add_one(x):
    return x + 1


class TestParallelLoad:
    """Test parallel_load() for multi-file I/O."""

    @pytest.mark.smoke
    def test_parallel_load_empty_list(self):
        from rheojax.parallel.api import parallel_load

        results = parallel_load([])
        assert results == []

    def test_parallel_load_returns_list_of_rheodata(self, tmp_path):
        from rheojax.parallel.api import parallel_load

        # Create minimal CSV files
        for i in range(3):
            f = tmp_path / f"data_{i}.csv"
            f.write_text("time,stress\n0.1,100\n1.0,50\n10.0,10\n")
        files = sorted(tmp_path.glob("*.csv"))
        results = parallel_load(files, x_col="time", y_col="stress")
        assert len(results) == 3
        for r in results:
            assert hasattr(r, "x")
            assert hasattr(r, "y")

    def test_parallel_load_sequential_fallback(self, tmp_path):
        from rheojax.parallel.api import parallel_load

        for i in range(2):
            f = tmp_path / f"data_{i}.csv"
            f.write_text("time,stress\n0.1,100\n1.0,50\n")
        files = sorted(tmp_path.glob("*.csv"))
        with patch.dict("os.environ", {"RHEOJAX_SEQUENTIAL": "1"}):
            results = parallel_load(files, x_col="time", y_col="stress")
        assert len(results) == 2


class TestParallelMap:
    """Test parallel_map() generic fan-out."""

    @pytest.mark.smoke
    def test_parallel_map_sequential_fallback(self):
        from rheojax.parallel.api import parallel_map

        with patch.dict("os.environ", {"RHEOJAX_SEQUENTIAL": "1"}):
            results = list(parallel_map(_test_add_one, [1, 2, 3]))
            assert sorted(results) == [2, 3, 4]

    def test_parallel_map_with_workers(self):
        from rheojax.parallel.api import parallel_map

        results = list(parallel_map(_test_add_one, range(10), n_workers=2))
        assert sorted(results) == list(range(1, 11))

    def test_parallel_map_empty(self):
        from rheojax.parallel.api import parallel_map

        results = list(parallel_map(_test_add_one, []))
        assert results == []
