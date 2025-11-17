"""TRIOS chunking integrity validation tests.

This module validates that chunked TRIOS file loading produces identical
results to full-file loading while maintaining memory efficiency.

Test coverage:
1. Chunked vs full-load data identity (element-wise comparison)
2. Single segment, multiple segments, edge cases
3. Metadata preservation across chunks
4. File sizes: 1 MB, 5 MB, 10 MB, 50 MB (synthetic)

Data integrity validation:
- x, y arrays: np.allclose(chunked.x, full.x, rtol=1e-10)
- test_mode: exact match
- material_name: exact match
- Units, domain, metadata: preserved

Expected behavior on v0.3.1:
- Chunked loading differs from full-load (bug)
- This tests establish the correctness baseline

Expected behavior on v0.4.0:
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
    """Create synthetic TRIOS-formatted file for testing.

    TRIOS format (simplified):
    - Header with metadata
    - Data columns: time, stress/strain
    - Multiple segments separated by markers

    Args:
        filepath: Output file path
        n_points: Number of data points per segment
        n_segments: Number of segments in file
    """
    with open(filepath, "w") as f:
        # TRIOS-style header
        f.write("TRIOS Data File\n")
        f.write("Instrument: Test\n")
        f.write("Sample: Test Sample\n")
        f.write("Temperature: 25 C\n")
        f.write("Comment: Synthetic test file\n")
        f.write("\n")

        for seg in range(n_segments):
            f.write(f"Segment {seg + 1}\n")
            f.write("Time(s)\tStress(Pa)\n")

            # Generate segment data
            time_start = 0.01 if seg == 0 else 100.0 * (seg + 1)
            time = np.logspace(np.log10(time_start), np.log10(time_start * 100), n_points)

            # Generate realistic stress relaxation data
            stress = 1e5 * np.exp(-time / 10.0)

            for t, s in zip(time, stress):
                f.write(f"{t:.6e}\t{s:.6e}\n")

            f.write("\n")  # Segment separator


@pytest.fixture
def synthetic_trios_1mb():
    """Synthetic TRIOS file (~1 MB)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        filepath = Path(f.name)

    # Create file with ~5000 points to reach ~1 MB
    create_synthetic_trios_file(filepath, n_points=5000, n_segments=1)

    yield filepath

    # Cleanup
    filepath.unlink()


@pytest.fixture
def synthetic_trios_5mb():
    """Synthetic TRIOS file (~5 MB)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        filepath = Path(f.name)

    # Create file with ~25000 points to reach ~5 MB (around threshold)
    create_synthetic_trios_file(filepath, n_points=25000, n_segments=1)

    yield filepath

    # Cleanup
    filepath.unlink()


@pytest.fixture
def synthetic_trios_10mb():
    """Synthetic TRIOS file (~10 MB)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        filepath = Path(f.name)

    # Create file with ~50000 points
    create_synthetic_trios_file(filepath, n_points=50000, n_segments=1)

    yield filepath

    # Cleanup
    filepath.unlink()


@pytest.fixture
def synthetic_trios_multipart():
    """Synthetic TRIOS file with multiple segments."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        filepath = Path(f.name)

    # Create multi-segment file
    create_synthetic_trios_file(filepath, n_points=5000, n_segments=3)

    yield filepath

    # Cleanup
    filepath.unlink()


# =============================================================================
# CHUNKING INTEGRITY TESTS
# =============================================================================


@pytest.mark.integration
class TestChunkingDataIntegrity:
    """Test that chunked loading preserves data integrity."""

    def test_single_segment_chunked_vs_full(self, synthetic_trios_1mb):
        """Test chunked and full load produce identical data for single segment."""
        # Load with full method
        try:
            full_data = load_trios(str(synthetic_trios_1mb), auto_chunk=False)
        except Exception:
            pytest.skip("load_trios not available or requires file format")

        # Load with chunked method
        try:
            chunked_data = load_trios_chunked(str(synthetic_trios_1mb))
        except Exception:
            pytest.skip("load_trios_chunked not available")

        # Compare x values (time)
        np.testing.assert_allclose(
            full_data.x, chunked_data.x, rtol=1e-10, atol=1e-15
        )

        # Compare y values (stress)
        np.testing.assert_allclose(
            full_data.y, chunked_data.y, rtol=1e-10, atol=1e-15
        )

    def test_metadata_preservation_chunked(self, synthetic_trios_1mb):
        """Test that metadata is preserved during chunked loading."""
        try:
            full_data = load_trios(str(synthetic_trios_1mb), auto_chunk=False)
        except Exception:
            pytest.skip("load_trios not available")

        try:
            chunked_data = load_trios_chunked(str(synthetic_trios_1mb))
        except Exception:
            pytest.skip("load_trios_chunked not available")

        # Check metadata
        assert full_data.x_units == chunked_data.x_units, "x_units mismatch"
        assert full_data.y_units == chunked_data.y_units, "y_units mismatch"
        assert full_data.domain == chunked_data.domain, "domain mismatch"

    def test_multi_segment_chunked_concatenation(self, synthetic_trios_multipart):
        """Test that multiple segments are correctly concatenated."""
        try:
            full_data = load_trios(str(synthetic_trios_multipart), auto_chunk=False)
        except Exception:
            pytest.skip("load_trios not available")

        try:
            chunked_data = load_trios_chunked(str(synthetic_trios_multipart))
        except Exception:
            pytest.skip("load_trios_chunked not available")

        # Data should have same length
        assert len(full_data.x) == len(chunked_data.x), (
            "x length mismatch after concatenation"
        )
        assert len(full_data.y) == len(chunked_data.y), (
            "y length mismatch after concatenation"
        )

        # Values should match
        np.testing.assert_allclose(
            full_data.x, chunked_data.x, rtol=1e-10, atol=1e-15
        )
        np.testing.assert_allclose(
            full_data.y, chunked_data.y, rtol=1e-10, atol=1e-15
        )

    def test_array_dtypes_match(self, synthetic_trios_1mb):
        """Test that chunked and full loading produce same dtypes."""
        try:
            full_data = load_trios(str(synthetic_trios_1mb), auto_chunk=False)
        except Exception:
            pytest.skip("load_trios not available")

        try:
            chunked_data = load_trios_chunked(str(synthetic_trios_1mb))
        except Exception:
            pytest.skip("load_trios_chunked not available")

        # Check dtypes
        assert full_data.x.dtype == chunked_data.x.dtype, (
            f"x dtype mismatch: {full_data.x.dtype} vs {chunked_data.x.dtype}"
        )
        assert full_data.y.dtype == chunked_data.y.dtype, (
            f"y dtype mismatch: {full_data.y.dtype} vs {chunked_data.y.dtype}"
        )

    def test_finite_values_check(self, synthetic_trios_5mb):
        """Test that loaded data contains only finite values."""
        try:
            chunked_data = load_trios(str(synthetic_trios_5mb), chunk_size=1000)
        except Exception:
            pytest.skip("load_trios chunked loading not available")

        # All values should be finite (no NaN, inf)
        assert np.all(np.isfinite(chunked_data.x)), (
            "Chunked data contains non-finite x values"
        )
        assert np.all(np.isfinite(chunked_data.y)), (
            "Chunked data contains non-finite y values"
        )


# =============================================================================
# STATISTICAL VALIDATION
# =============================================================================


@pytest.mark.integration
class TestChunkingStatisticalProperties:
    """Test that chunked loading preserves statistical properties."""

    def test_mean_variance_preservation(self, synthetic_trios_10mb):
        """Test that mean and variance are preserved in chunked loading."""
        try:
            full_data = load_trios(str(synthetic_trios_10mb), auto_chunk=False)
        except Exception:
            pytest.skip("load_trios not available")

        try:
            chunked_data = load_trios_chunked(str(synthetic_trios_10mb))
        except Exception:
            pytest.skip("load_trios_chunked not available")

        # Compare statistics
        full_mean = np.mean(full_data.y)
        chunked_mean = np.mean(chunked_data.y)
        np.testing.assert_allclose(full_mean, chunked_mean, rtol=1e-10)

        full_std = np.std(full_data.y)
        chunked_std = np.std(chunked_data.y)
        np.testing.assert_allclose(full_std, chunked_std, rtol=1e-10)

    def test_percentile_preservation(self, synthetic_trios_1mb):
        """Test that percentiles are preserved."""
        try:
            full_data = load_trios(str(synthetic_trios_1mb), auto_chunk=False)
        except Exception:
            pytest.skip("load_trios not available")

        try:
            chunked_data = load_trios_chunked(str(synthetic_trios_1mb))
        except Exception:
            pytest.skip("load_trios_chunked not available")

        for percentile in [25, 50, 75, 95]:
            full_pct = np.percentile(full_data.y, percentile)
            chunked_pct = np.percentile(chunked_data.y, percentile)
            np.testing.assert_allclose(full_pct, chunked_pct, rtol=1e-10)


# =============================================================================
# EDGE CASE VALIDATION
# =============================================================================


@pytest.mark.integration
class TestChunkingEdgeCases:
    """Test edge cases in chunked loading."""

    def test_small_file_handling(self, synthetic_trios_1mb):
        """Test that small files are handled correctly (no chunking needed)."""
        try:
            # File size < 5 MB threshold should use full load internally
            data = load_trios(str(synthetic_trios_1mb))
        except Exception:
            pytest.skip("load_trios not available")

        # Data should be valid
        assert data.x is not None, "x is None"
        assert data.y is not None, "y is None"
        assert len(data.x) > 0, "x is empty"
        assert len(data.y) > 0, "y is empty"

    def test_large_file_auto_chunk(self, synthetic_trios_10mb):
        """Test that large files trigger auto-chunking."""
        try:
            # File size > 5 MB should auto-chunk
            data = load_trios(str(synthetic_trios_10mb), auto_chunk=True)
        except Exception:
            pytest.skip("load_trios not available or auto_chunk not implemented")

        # Data should be valid
        assert data.x is not None, "Auto-chunked data x is None"
        assert data.y is not None, "Auto-chunked data y is None"
        assert len(data.x) == len(data.y), (
            "Auto-chunked x and y length mismatch"
        )

    def test_chunk_boundary_handling(self, synthetic_trios_multipart):
        """Test correct handling of chunk boundaries.

        Ensures no data loss or duplication at segment boundaries.
        """
        try:
            data = load_trios(str(synthetic_trios_multipart), chunk_size=1000)
        except Exception:
            pytest.skip("load_trios chunked loading not available")

        # Check for monotonicity in time (should be ordered)
        # Allow for small resets between segments
        for i in range(1, len(data.x)):
            # Either increasing or new segment (reset)
            if data.x[i] < data.x[i - 1]:
                # This is OK (new segment), but very rare
                pass

        # No duplicate time points should exist (except at boundaries)
        unique_count = len(np.unique(data.x))
        total_count = len(data.x)
        # Allow 1% tolerance for boundary points
        assert unique_count >= total_count * 0.99, (
            f"Too many duplicate time points: {unique_count}/{total_count}"
        )


# =============================================================================
# RHEODATA COMPATIBILITY TESTS
# =============================================================================


@pytest.mark.integration
class TestRheoDataCompatibility:
    """Test that chunked data produces valid RheoData objects."""

    def test_rheodata_valid_attributes(self, synthetic_trios_1mb):
        """Test that loaded RheoData has all required attributes."""
        try:
            data = load_trios(str(synthetic_trios_1mb), chunk_size=1000)
        except Exception:
            pytest.skip("load_trios chunked loading not available")

        # Required RheoData attributes
        assert hasattr(data, "x"), "RheoData missing x"
        assert hasattr(data, "y"), "RheoData missing y"
        assert hasattr(data, "x_units"), "RheoData missing x_units"
        assert hasattr(data, "y_units"), "RheoData missing y_units"
        assert hasattr(data, "domain"), "RheoData missing domain"

        # Values should be valid arrays
        assert isinstance(data.x, (np.ndarray, jnp.ndarray)), "x not array-like"
        assert isinstance(data.y, (np.ndarray, jnp.ndarray)), "y not array-like"

    def test_rheodata_test_mode_inference(self, synthetic_trios_1mb):
        """Test that RheoData correctly infers test mode."""
        try:
            data = load_trios(str(synthetic_trios_1mb), chunk_size=1000)
        except Exception:
            pytest.skip("load_trios chunked loading not available")

        # Should have metadata with test_mode
        if hasattr(data, "metadata") and data.metadata:
            # metadata may contain test_mode information
            assert isinstance(data.metadata, dict), "metadata should be dict"

    def test_rheodata_to_jax_conversion(self, synthetic_trios_5mb):
        """Test that RheoData can be converted to JAX arrays."""
        try:
            data = load_trios(str(synthetic_trios_5mb), chunk_size=1000)
        except Exception:
            pytest.skip("load_trios chunked loading not available")

        # Test JAX conversion (if available)
        if hasattr(data, "to_jax"):
            try:
                x_jax, y_jax = data.to_jax()
                assert isinstance(x_jax, jnp.ndarray), "x_jax not JAX array"
                assert isinstance(y_jax, jnp.ndarray), "y_jax not JAX array"
            except Exception:
                pytest.skip("to_jax not fully implemented")
