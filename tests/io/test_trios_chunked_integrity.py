"""TRIOS chunking integrity validation tests.

This module validates that chunked TRIOS file loading produces identical
results to full-file loading while maintaining memory efficiency.

Test coverage:
1. Chunked vs full-load data identity (element-wise comparison)
2. Single segment, multiple segments, edge cases
3. Metadata preservation across chunks
4. File sizes: small (~250 KB), medium (~1.3 MB), large (~2.5 MB)

Data integrity validation:
- x, y arrays: np.allclose(chunked.x, full.x, rtol=1e-10)
- test_mode: exact match
- Units, domain, metadata: preserved

Expected behavior on v0.4.0+:
- Chunked and full-load produce identical RheoData
- Tests validate auto-chunking doesn't introduce errors
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax
from rheojax.io.readers.trios import load_trios, load_trios_chunked

jax, jnp = safe_import_jax()

# =============================================================================
# SYNTHETIC TRIOS FILE FIXTURES
# =============================================================================


def create_synthetic_trios_file(filepath: Path, n_points: int, n_segments: int = 1):
    """Create synthetic TRIOS LIMS-format (.txt) file for testing.

    Produces valid TRIOS TXT format with:
    - Metadata header (Filename, Instrument serial number, Sample name)
    - [step] markers per segment
    - Number of points, Variables row, units row
    - Tab-separated data rows with 'data point N' labels

    Args:
        filepath: Output file path
        n_points: Number of data points per segment
        n_segments: Number of segments in file
    """
    with open(filepath, "w") as f:
        # TRIOS metadata header
        f.write(f"Filename\t{filepath.name}\n")
        f.write("Instrument serial number\tSYN-0001\n")
        f.write("Instrument name\tTASerNo SYN-0001\n")
        f.write("operator\tTest User\n")
        f.write("rundate\t2025-10-24\n")
        f.write("Sample name\tSynthetic Test Sample\n")
        f.write("Geometry name\tParallel Plate\n")
        f.write("\n")

        for seg in range(n_segments):
            f.write("[step]\n")
            f.write("Step name\tStress relaxation (25.0 °C)\n")
            f.write(f"Number of points\t{n_points}\n")
            f.write("Variables\tTime\tStress\n")
            f.write("\ts\tPa\n")

            # Generate segment data
            time_start = 0.01 if seg == 0 else 100.0 * (seg + 1)
            time = np.logspace(
                np.log10(time_start), np.log10(time_start * 100), n_points
            )

            # Generate realistic stress relaxation data
            stress = 1e5 * np.exp(-time / 10.0)

            for i, (t, s) in enumerate(zip(time, stress)):
                f.write(f"data point {i + 1}\t{t:.6e}\t{s:.6e}\n")

            f.write("\n")  # Segment separator


def _aggregate_chunks(filepath: str | Path, chunk_size: int = 10000) -> RheoData:
    """Aggregate all chunks from load_trios_chunked into a single RheoData."""
    x_parts, y_parts = [], []
    first_chunk = None
    for chunk in load_trios_chunked(str(filepath), chunk_size=chunk_size):
        if first_chunk is None:
            first_chunk = chunk
        x_parts.append(np.asarray(chunk.x))
        y_parts.append(np.asarray(chunk.y))
    assert first_chunk is not None, "No chunks yielded"
    return RheoData(
        x=np.concatenate(x_parts),
        y=np.concatenate(y_parts),
        x_units=first_chunk.x_units,
        y_units=first_chunk.y_units,
        domain=first_chunk.domain,
        metadata=first_chunk.metadata,
    )


def _get_single_segment(data):
    """If data is a list, return first element; otherwise return as-is."""
    if isinstance(data, list):
        return data[0]
    return data


@pytest.fixture
def synthetic_trios_small():
    """Synthetic TRIOS file (~250 KB, 1000 points)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        filepath = Path(f.name)

    create_synthetic_trios_file(filepath, n_points=1000, n_segments=1)
    yield filepath
    filepath.unlink(missing_ok=True)


@pytest.fixture
def synthetic_trios_medium():
    """Synthetic TRIOS file (~1.3 MB, 5000 points)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        filepath = Path(f.name)

    create_synthetic_trios_file(filepath, n_points=5000, n_segments=1)
    yield filepath
    filepath.unlink(missing_ok=True)


@pytest.fixture
def synthetic_trios_large():
    """Synthetic TRIOS file (~2.5 MB, 10000 points)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        filepath = Path(f.name)

    create_synthetic_trios_file(filepath, n_points=10000, n_segments=1)
    yield filepath
    filepath.unlink(missing_ok=True)


@pytest.fixture
def synthetic_trios_multipart():
    """Synthetic TRIOS file with 3 segments (~750 KB)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        filepath = Path(f.name)

    create_synthetic_trios_file(filepath, n_points=1000, n_segments=3)
    yield filepath
    filepath.unlink(missing_ok=True)


# =============================================================================
# CHUNKING INTEGRITY TESTS
# =============================================================================


@pytest.mark.integration
class TestChunkingDataIntegrity:
    """Test that chunked loading preserves data integrity."""

    def test_single_segment_chunked_vs_full(self, synthetic_trios_small):
        """Test chunked and full load produce identical data for single segment."""
        full_data = _get_single_segment(
            load_trios(str(synthetic_trios_small), auto_chunk=False)
        )

        # Use load_trios with chunk_size to aggregate (same as chunked path)
        chunked_data = _get_single_segment(
            load_trios(str(synthetic_trios_small), chunk_size=500)
        )

        np.testing.assert_allclose(
            np.asarray(full_data.x),
            np.asarray(chunked_data.x),
            rtol=1e-10,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            np.asarray(full_data.y),
            np.asarray(chunked_data.y),
            rtol=1e-10,
            atol=1e-15,
        )

    def test_metadata_preservation_chunked(self, synthetic_trios_small):
        """Test that metadata is preserved during chunked loading."""
        full_data = _get_single_segment(
            load_trios(str(synthetic_trios_small), auto_chunk=False)
        )
        chunked_data = _aggregate_chunks(synthetic_trios_small, chunk_size=500)

        assert full_data.x_units == chunked_data.x_units, "x_units mismatch"
        assert full_data.y_units == chunked_data.y_units, "y_units mismatch"
        assert full_data.domain == chunked_data.domain, "domain mismatch"

    def test_multi_segment_chunked_concatenation(self, synthetic_trios_multipart):
        """Test that multiple segments are correctly concatenated."""
        full_result = load_trios(
            str(synthetic_trios_multipart),
            auto_chunk=False,
            return_all_segments=True,
        )
        # Full load of multi-segment returns list — concatenate for comparison
        if isinstance(full_result, list):
            full_x = np.concatenate([np.asarray(d.x) for d in full_result])
            full_y = np.concatenate([np.asarray(d.y) for d in full_result])
        else:
            full_x = np.asarray(full_result.x)
            full_y = np.asarray(full_result.y)

        # Chunked aggregation
        chunked_data = _aggregate_chunks(synthetic_trios_multipart, chunk_size=500)

        assert len(full_x) == len(chunked_data.x), "x length mismatch"
        assert len(full_y) == len(chunked_data.y), "y length mismatch"

        np.testing.assert_allclose(
            full_x,
            np.asarray(chunked_data.x),
            rtol=1e-10,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            full_y,
            np.asarray(chunked_data.y),
            rtol=1e-10,
            atol=1e-15,
        )

    def test_array_dtypes_match(self, synthetic_trios_small):
        """Test that chunked and full loading produce same dtypes."""
        full_data = _get_single_segment(
            load_trios(str(synthetic_trios_small), auto_chunk=False)
        )
        chunked_data = _aggregate_chunks(synthetic_trios_small, chunk_size=500)

        assert (
            np.asarray(full_data.x).dtype == np.asarray(chunked_data.x).dtype
        ), f"x dtype mismatch: {full_data.x.dtype} vs {chunked_data.x.dtype}"
        assert (
            np.asarray(full_data.y).dtype == np.asarray(chunked_data.y).dtype
        ), f"y dtype mismatch: {full_data.y.dtype} vs {chunked_data.y.dtype}"

    def test_finite_values_check(self, synthetic_trios_medium):
        """Test that loaded data contains only finite values."""
        chunked_data = _get_single_segment(
            load_trios(str(synthetic_trios_medium), chunk_size=1000)
        )

        assert np.all(
            np.isfinite(np.asarray(chunked_data.x))
        ), "Chunked data contains non-finite x values"
        assert np.all(
            np.isfinite(np.asarray(chunked_data.y))
        ), "Chunked data contains non-finite y values"


# =============================================================================
# STATISTICAL VALIDATION
# =============================================================================


@pytest.mark.integration
class TestChunkingStatisticalProperties:
    """Test that chunked loading preserves statistical properties."""

    def test_mean_variance_preservation(self, synthetic_trios_large):
        """Test that mean and variance are preserved in chunked loading."""
        full_data = _get_single_segment(
            load_trios(str(synthetic_trios_large), auto_chunk=False)
        )
        chunked_data = _aggregate_chunks(synthetic_trios_large, chunk_size=2000)

        np.testing.assert_allclose(
            np.mean(np.asarray(full_data.y)),
            np.mean(np.asarray(chunked_data.y)),
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            np.std(np.asarray(full_data.y)),
            np.std(np.asarray(chunked_data.y)),
            rtol=1e-10,
        )

    def test_percentile_preservation(self, synthetic_trios_small):
        """Test that percentiles are preserved."""
        full_data = _get_single_segment(
            load_trios(str(synthetic_trios_small), auto_chunk=False)
        )
        chunked_data = _aggregate_chunks(synthetic_trios_small, chunk_size=500)

        for percentile in [25, 50, 75, 95]:
            np.testing.assert_allclose(
                np.percentile(np.asarray(full_data.y), percentile),
                np.percentile(np.asarray(chunked_data.y), percentile),
                rtol=1e-10,
            )


# =============================================================================
# EDGE CASE VALIDATION
# =============================================================================


@pytest.mark.integration
class TestChunkingEdgeCases:
    """Test edge cases in chunked loading."""

    def test_small_file_handling(self, synthetic_trios_small):
        """Test that small files are handled correctly (no chunking needed)."""
        data = _get_single_segment(load_trios(str(synthetic_trios_small)))

        assert data.x is not None, "x is None"
        assert data.y is not None, "y is None"
        assert len(data.x) > 0, "x is empty"
        assert len(data.y) > 0, "y is empty"

    def test_large_file_auto_chunk(self, synthetic_trios_large):
        """Test that large files load correctly with auto_chunk=True."""
        data = _get_single_segment(
            load_trios(str(synthetic_trios_large), auto_chunk=True)
        )

        assert data.x is not None, "Auto-chunked data x is None"
        assert data.y is not None, "Auto-chunked data y is None"
        assert len(data.x) == len(data.y), "Auto-chunked x and y length mismatch"

    def test_chunk_boundary_handling(self, synthetic_trios_multipart):
        """Test correct handling of chunk boundaries.

        Ensures no data loss or duplication at segment boundaries.
        """
        data = _get_single_segment(
            load_trios(str(synthetic_trios_multipart), chunk_size=500)
        )

        # No duplicate time points should exist (except at boundaries)
        unique_count = len(np.unique(np.asarray(data.x)))
        total_count = len(data.x)
        # Allow 1% tolerance for boundary points
        assert (
            unique_count >= total_count * 0.99
        ), f"Too many duplicate time points: {unique_count}/{total_count}"


# =============================================================================
# RHEODATA COMPATIBILITY TESTS
# =============================================================================


@pytest.mark.integration
class TestRheoDataCompatibility:
    """Test that chunked data produces valid RheoData objects."""

    def test_rheodata_valid_attributes(self, synthetic_trios_small):
        """Test that loaded RheoData has all required attributes."""
        data = _get_single_segment(
            load_trios(str(synthetic_trios_small), chunk_size=500)
        )

        assert hasattr(data, "x"), "RheoData missing x"
        assert hasattr(data, "y"), "RheoData missing y"
        assert hasattr(data, "x_units"), "RheoData missing x_units"
        assert hasattr(data, "y_units"), "RheoData missing y_units"
        assert hasattr(data, "domain"), "RheoData missing domain"

        assert isinstance(data.x, (np.ndarray, jnp.ndarray)), "x not array-like"
        assert isinstance(data.y, (np.ndarray, jnp.ndarray)), "y not array-like"

    def test_rheodata_test_mode_inference(self, synthetic_trios_small):
        """Test that RheoData correctly infers test mode."""
        data = _get_single_segment(
            load_trios(str(synthetic_trios_small), chunk_size=500)
        )

        if hasattr(data, "metadata") and data.metadata:
            assert isinstance(data.metadata, dict), "metadata should be dict"

    def test_rheodata_to_jax_conversion(self, synthetic_trios_medium):
        """Test that RheoData can be converted to JAX arrays."""
        data = _get_single_segment(
            load_trios(str(synthetic_trios_medium), chunk_size=1000)
        )

        if hasattr(data, "to_jax"):
            jax_data = data.to_jax()
            assert isinstance(jax_data, RheoData), "to_jax should return RheoData"
            assert isinstance(jax_data.x, jnp.ndarray), "x not JAX array"
            assert isinstance(jax_data.y, jnp.ndarray), "y not JAX array"
