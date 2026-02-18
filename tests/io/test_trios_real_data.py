"""Integration tests for TRIOS reader using real OWChirp data files.

These tests verify the TRIOS reader against actual TRIOS-exported files
from TA Instruments rheometers (anonymized).
"""

from pathlib import Path

import numpy as np
import pytest

from rheojax.core.data import RheoData

# Real TRIOS data files (anonymized, shipped with examples)
LAOS_DIR = Path(__file__).parent.parent.parent / "examples" / "data" / "laos"
TRIOS_TCS = LAOS_DIR / "owchirp_tcs.txt"
TRIOS_TTS = LAOS_DIR / "owchirp_tts.txt"

HAS_TRIOS_DATA = TRIOS_TCS.exists() and TRIOS_TTS.exists()
skip_no_data = pytest.mark.skipif(
    not HAS_TRIOS_DATA,
    reason="Real TRIOS data files not found (examples/data/laos/owchirp_*.txt)",
)


@skip_no_data
class TestTriosRealDataLoading:
    """Test TRIOS reader against real instrument-exported files."""

    def test_load_owchirp_tcs(self):
        """Load OWChirp time-concentration-superposition file."""
        from rheojax.io.readers.trios import load_trios

        data = load_trios(str(TRIOS_TCS))
        assert isinstance(data, (RheoData, list))
        if isinstance(data, list):
            for d in data:
                assert isinstance(d, RheoData)
                assert len(d.x) > 0
        else:
            assert len(data.x) > 0

    def test_load_owchirp_tts(self):
        """Load OWChirp time-temperature-superposition file."""
        from rheojax.io.readers.trios import load_trios

        data = load_trios(str(TRIOS_TTS))
        assert isinstance(data, (RheoData, list))
        if isinstance(data, list):
            for d in data:
                assert isinstance(d, RheoData)
                assert len(d.x) > 0
        else:
            assert len(data.x) > 0

    def test_tcs_data_integrity(self):
        """Verify TCS file produces finite, non-empty arrays."""
        from rheojax.io.readers.trios import load_trios

        data = load_trios(str(TRIOS_TCS))
        if isinstance(data, list):
            data = data[0]

        # Data should have thousands of points
        assert len(data.x) > 1000, f"Expected >1000 points, got {len(data.x)}"

        # All values must be finite
        assert np.all(np.isfinite(data.x)), "x contains non-finite values"
        if np.isrealobj(data.y):
            assert np.all(np.isfinite(data.y)), "y contains non-finite values"
        else:
            assert np.all(np.isfinite(data.y.real)), "y.real contains non-finite"
            assert np.all(np.isfinite(data.y.imag)), "y.imag contains non-finite"

    def test_tts_data_integrity(self):
        """Verify TTS file produces finite, non-empty arrays."""
        from rheojax.io.readers.trios import load_trios

        data = load_trios(str(TRIOS_TTS))
        if isinstance(data, list):
            data = data[0]

        assert len(data.x) > 1000, f"Expected >1000 points, got {len(data.x)}"
        assert np.all(np.isfinite(data.x)), "x contains non-finite values"

    def test_auto_load_detects_trios(self):
        """Verify auto_load correctly routes .txt TRIOS files."""
        from rheojax.io.readers.auto import auto_load

        data = auto_load(str(TRIOS_TCS))
        assert isinstance(data, (RheoData, list))

    def test_domain_detected(self):
        """Verify domain is auto-detected for TRIOS data."""
        from rheojax.io.readers.trios import load_trios

        data = load_trios(str(TRIOS_TCS))
        if isinstance(data, list):
            data = data[0]

        assert data.domain in ("time", "frequency"), f"Unexpected domain: {data.domain}"

    def test_units_present(self):
        """Verify units are extracted from TRIOS headers."""
        from rheojax.io.readers.trios import load_trios

        data = load_trios(str(TRIOS_TCS))
        if isinstance(data, list):
            data = data[0]

        # TRIOS files should have units in headers
        assert data.x_units is not None, "x_units not detected"

    def test_metadata_extracted(self):
        """Verify metadata is extracted from TRIOS headers."""
        from rheojax.io.readers.trios import load_trios

        data = load_trios(str(TRIOS_TCS))
        if isinstance(data, list):
            data = data[0]

        assert data.metadata is not None
        assert len(data.metadata) > 0

    def test_both_files_load_without_errors(self):
        """Regression: both TRIOS files load without any exceptions."""
        from rheojax.io.readers.trios import load_trios

        # This is a catch-all: no exception = pass
        load_trios(str(TRIOS_TCS))
        load_trios(str(TRIOS_TTS))


@skip_no_data
class TestTriosRealDataChunked:
    """Test chunked loading of real TRIOS files."""

    def test_chunked_matches_full_load(self):
        """Chunked loading should produce identical data to full loading."""
        from rheojax.io.readers.trios import load_trios

        # Full load
        full = load_trios(str(TRIOS_TCS))
        if isinstance(full, list):
            full = full[0]

        # Chunked load (if supported)
        try:
            chunked = load_trios(str(TRIOS_TCS), auto_chunk=True)
        except TypeError:
            pytest.skip("Chunked loading not supported by this TRIOS reader version")
            return

        if isinstance(chunked, list):
            chunked = chunked[0]

        np.testing.assert_array_equal(full.x, chunked.x)
        np.testing.assert_array_equal(full.y, chunked.y)
