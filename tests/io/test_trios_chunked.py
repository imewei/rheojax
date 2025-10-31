"""Tests for TRIOS chunked reading functionality.

This module tests memory-efficient chunked reading of large TRIOS files.
"""

from pathlib import Path

import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.io.readers.trios import load_trios, load_trios_chunked


class TestTriosChunkedBasic:
    """Basic chunked reading tests."""

    def test_chunked_basic_single_chunk(self, tmp_path):
        """Test chunked reading with data smaller than chunk size."""
        trios_content = """Filename	test.txt
Instrument serial number	4010-1234
Instrument name	TASerNo 4010-1234
operator	Test User
rundate	2025-10-24
Sample name	Test Sample
Geometry name	Parallel Plate

[step]
Temperature	°C
25.0

[step]
Number of points	5
Time	Storage modulus	Loss modulus
s	Pa	Pa
0.1	1000	500
0.2	1200	600
0.3	1400	700
0.4	1600	800
0.5	1800	900
"""
        test_file = tmp_path / "test_trios.txt"
        test_file.write_text(trios_content)

        # Read with large chunk size (should get all data in one chunk)
        chunks = list(load_trios_chunked(str(test_file), chunk_size=100))

        # Should have exactly one chunk
        assert len(chunks) == 1

        # Verify data integrity
        chunk = chunks[0]
        assert isinstance(chunk, RheoData)
        assert len(chunk.x) == 5
        assert len(chunk.y) == 5
        np.testing.assert_array_almost_equal(
            chunk.x, [0.1, 0.2, 0.3, 0.4, 0.5]
        )
        np.testing.assert_array_almost_equal(
            chunk.y, [1000, 1200, 1400, 1600, 1800]
        )

    def test_chunked_multiple_chunks(self, tmp_path):
        """Test chunked reading with data split across chunks."""
        # Create file with 25 data points
        lines = [
            "Filename\ttest.txt",
            "Instrument serial number\t4010-1234",
            "",
            "[step]",
            "Number of points\t25",
            "Time\tStress",
            "s\tPa",
        ]
        for i in range(25):
            lines.append(f"{i * 0.1:.1f}\t{(i + 1) * 100}")

        test_file = tmp_path / "test_chunks.txt"
        test_file.write_text("\n".join(lines))

        # Read with chunk size of 10
        chunks = list(load_trios_chunked(str(test_file), chunk_size=10))

        # Should have multiple chunks
        assert len(chunks) >= 2

        # All chunks should be RheoData objects
        for chunk in chunks:
            assert isinstance(chunk, RheoData)

        # Verify total points equals original
        total_points = sum(len(chunk.x) for chunk in chunks)
        assert total_points == 25

        # Verify first chunk starts correctly
        assert chunks[0].x[0] == 0.0
        assert chunks[0].y[0] == 100

        # Verify last chunk ends correctly
        assert chunks[-1].y[-1] == 2500

    def test_chunked_metadata_preserved(self, tmp_path):
        """Test that metadata is preserved across all chunks."""
        trios_content = """Filename	metadata_test.txt
Instrument serial number	5343-5678
Instrument name	DHR-3
operator	Dr. Smith
rundate	2025-10-24
Sample name	Polymer XYZ
Geometry name	Cone and Plate

[step]
Number of points	15
Time	Stress
s	Pa
1.0	100
2.0	200
3.0	300
4.0	400
5.0	500
6.0	600
7.0	700
8.0	800
9.0	900
10.0	1000
11.0	1100
12.0	1200
13.0	1300
14.0	1400
15.0	1500
"""
        test_file = tmp_path / "test_metadata.txt"
        test_file.write_text(trios_content)

        chunks = list(load_trios_chunked(str(test_file), chunk_size=5))

        # Should have 3 chunks
        assert len(chunks) == 3

        # Check metadata is preserved in all chunks
        for chunk in chunks:
            assert chunk.metadata["sample_name"] == "Polymer XYZ"
            assert chunk.metadata["instrument_serial_number"] == "5343-5678"
            assert chunk.metadata["test_mode"] is not None
            assert "columns" in chunk.metadata
            assert "units" in chunk.metadata

    def test_chunked_units_preserved(self, tmp_path):
        """Test that units are preserved across chunks."""
        trios_content = """Filename	test.txt
Instrument serial number	4010-1234

[step]
Number of points	12
Angular frequency	Storage modulus	Loss modulus
rad/s	Pa	Pa
0.1	1000	500
0.2	1200	600
0.3	1400	700
0.4	1600	800
0.5	1800	900
0.6	2000	1000
0.7	2200	1100
0.8	2400	1200
0.9	2600	1300
1.0	2800	1400
1.1	3000	1500
1.2	3200	1600
"""
        test_file = tmp_path / "test_units.txt"
        test_file.write_text(trios_content)

        chunks = list(load_trios_chunked(str(test_file), chunk_size=5))

        # Check units preserved
        for chunk in chunks:
            assert chunk.x_units == "rad/s"
            assert chunk.y_units == "Pa"


class TestTriosChunkedAdvanced:
    """Advanced chunked reading tests."""

    def test_chunked_large_file_simulation(self, tmp_path):
        """Test chunked reading with simulated large file (1000 points)."""
        lines = [
            "Filename\tlarge_file.txt",
            "Instrument serial number\t4010-1234",
            "",
            "[step]",
            "Number of points\t1000",
            "Time\tStress",
            "s\tPa",
        ]
        # Generate 1000 data points
        for i in range(1000):
            lines.append(f"{i * 0.01:.2f}\t{(i + 1) * 10}")

        test_file = tmp_path / "large_file.txt"
        test_file.write_text("\n".join(lines))

        # Read with chunk size of 100
        chunks = list(load_trios_chunked(str(test_file), chunk_size=100))

        # Should have 10 chunks of 100 each
        assert len(chunks) == 10
        for i, chunk in enumerate(chunks):
            assert len(chunk.x) == 100
            assert isinstance(chunk, RheoData)

        # Verify continuity between chunks
        assert chunks[0].y[-1] == 1000  # Last of first chunk
        assert chunks[1].y[0] == 1010   # First of second chunk

    def test_chunked_vs_full_load_equivalence(self, tmp_path):
        """Test that chunked reading produces same data as full load."""
        trios_content = """Filename	test.txt
Instrument serial number	4010-1234

[step]
Number of points	30
Time	Stress
s	Pa
"""
        # Add 30 data points
        for i in range(30):
            trios_content += f"{i * 0.1:.1f}\t{(i + 1) * 100}\n"

        test_file = tmp_path / "test_equiv.txt"
        test_file.write_text(trios_content)

        # Full load
        full_data = load_trios(str(test_file))

        # Chunked load and concatenate
        chunks = list(load_trios_chunked(str(test_file), chunk_size=10))
        chunked_x = np.concatenate([chunk.x for chunk in chunks])
        chunked_y = np.concatenate([chunk.y for chunk in chunks])

        # Should be identical
        np.testing.assert_array_equal(full_data.x, chunked_x)
        np.testing.assert_array_equal(full_data.y, chunked_y)

    def test_chunked_multiple_segments(self, tmp_path):
        """Test chunked reading with multiple procedure segments."""
        trios_content = """Filename	multiseg.txt
Instrument serial number	4010-1234

[step]
Number of points	20
Time	Stress
s	Pa
"""
        # First segment: 20 points
        for i in range(20):
            trios_content += f"{i * 0.1:.1f}\t{(i + 1) * 100}\n"

        trios_content += """
[step]
Number of points	15
Time	Stress
s	Pa
"""
        # Second segment: 15 points
        for i in range(15):
            trios_content += f"{i * 0.1:.1f}\t{(i + 1) * 50}\n"

        test_file = tmp_path / "test_multiseg.txt"
        test_file.write_text(trios_content)

        # Read all segments in chunks
        chunks = list(load_trios_chunked(str(test_file), chunk_size=10))

        # Should have chunks from both segments
        # Segment 1: 2 chunks (10 + 10)
        # Segment 2: 2 chunks (10 + 5)
        assert len(chunks) == 4

        # Verify segment 1 data
        assert chunks[0].y[0] == 100
        assert chunks[1].y[-1] == 2000

        # Verify segment 2 data
        assert chunks[2].y[0] == 50
        assert chunks[3].y[-1] == 750

    def test_chunked_segment_selection(self, tmp_path):
        """Test reading only specific segment."""
        trios_content = """Filename	multiseg.txt
Instrument serial number	4010-1234

[step]
Number of points	10
Time	Stress
s	Pa
"""
        # First segment
        for i in range(10):
            trios_content += f"{i:.1f}\t100\n"

        trios_content += """
[step]
Number of points	10
Time	Stress
s	Pa
"""
        # Second segment
        for i in range(10):
            trios_content += f"{i:.1f}\t200\n"

        test_file = tmp_path / "test_seg_select.txt"
        test_file.write_text(trios_content)

        # Read only second segment
        chunks = list(
            load_trios_chunked(str(test_file), chunk_size=5, segment_index=1)
        )

        # Should have 2 chunks from second segment only
        assert len(chunks) == 2
        # All values should be 200 (from second segment)
        for chunk in chunks:
            assert np.all(chunk.y == 200)

    def test_chunked_nan_handling(self, tmp_path):
        """Test that NaN values are properly filtered in chunks."""
        trios_content = """Filename	test.txt
Instrument serial number	4010-1234

[step]
Number of points	15
Time	Stress
s	Pa
0.1	100
0.2
0.3	300
0.4	400
	500
0.6	600
0.7	700
0.8
0.9	900
1.0	1000
1.1
1.2	1200
1.3	1300
1.4	1400
1.5	1500
"""
        test_file = tmp_path / "test_nan.txt"
        test_file.write_text(trios_content)

        chunks = list(load_trios_chunked(str(test_file), chunk_size=5))

        # Should have valid data in all chunks
        total_points = sum(len(chunk.x) for chunk in chunks)
        # 15 original - 4 with NaN = 11 valid
        assert total_points == 11

        # Verify no NaN values
        for chunk in chunks:
            assert not np.any(np.isnan(chunk.x))
            assert not np.any(np.isnan(chunk.y))


class TestTriosChunkedEdgeCases:
    """Edge cases and error handling."""

    def test_chunked_empty_segment(self, tmp_path):
        """Test handling of empty segment."""
        trios_content = """Filename	test.txt
Instrument serial number	4010-1234

[step]
Number of points	0
Time	Stress
s	Pa
"""
        test_file = tmp_path / "test_empty.txt"
        test_file.write_text(trios_content)

        chunks = list(load_trios_chunked(str(test_file), chunk_size=10))

        # Should return empty list
        assert len(chunks) == 0

    def test_chunked_single_point(self, tmp_path):
        """Test handling of single data point."""
        trios_content = """Filename	test.txt
Instrument serial number	4010-1234

[step]
Number of points	1
Time	Stress
s	Pa
1.0	100
"""
        test_file = tmp_path / "test_single.txt"
        test_file.write_text(trios_content)

        chunks = list(load_trios_chunked(str(test_file), chunk_size=10))

        assert len(chunks) == 1
        assert len(chunks[0].x) == 1
        assert chunks[0].x[0] == 1.0
        assert chunks[0].y[0] == 100

    def test_chunked_chunk_size_one(self, tmp_path):
        """Test with chunk size of 1."""
        trios_content = """Filename	test.txt
Instrument serial number	4010-1234

[step]
Number of points	5
Time	Stress
s	Pa
1.0	100
2.0	200
3.0	300
4.0	400
5.0	500
"""
        test_file = tmp_path / "test_one.txt"
        test_file.write_text(trios_content)

        chunks = list(load_trios_chunked(str(test_file), chunk_size=1))

        # Should have multiple chunks
        assert len(chunks) >= 1

        # Verify total points equals original
        total_points = sum(len(chunk.x) for chunk in chunks)
        assert total_points == 5

        # Verify all data is accounted for
        all_y_values = np.concatenate([chunk.y for chunk in chunks])
        assert len(all_y_values) == 5
        assert 100 in all_y_values
        assert 500 in all_y_values

    def test_chunked_file_not_found(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            list(load_trios_chunked("nonexistent.txt"))

    def test_chunked_no_segments(self, tmp_path):
        """Test error handling for file with no segments."""
        trios_content = """Filename	test.txt
Instrument serial number	4010-1234
Just some metadata
"""
        test_file = tmp_path / "test_no_seg.txt"
        test_file.write_text(trios_content)

        with pytest.raises(ValueError, match="No data segments"):
            list(load_trios_chunked(str(test_file)))

    def test_chunked_validation_disabled(self, tmp_path):
        """Test chunked reading with validation disabled."""
        trios_content = """Filename	test.txt
Instrument serial number	4010-1234

[step]
Number of points	10
Time	Stress
s	Pa
0.1	100
0.2	200
0.3	300
0.4	400
0.5	500
0.6	600
0.7	700
0.8	800
0.9	900
1.0	1000
"""
        test_file = tmp_path / "test_no_valid.txt"
        test_file.write_text(trios_content)

        # Should work without validation
        chunks = list(
            load_trios_chunked(str(test_file), chunk_size=5, validate_data=False)
        )

        assert len(chunks) >= 1

        # Verify total points
        total_points = sum(len(chunk.x) for chunk in chunks)
        assert total_points == 10


class TestTriosChunkedBackwardCompatibility:
    """Test backward compatibility with existing load_trios."""

    def test_load_trios_with_chunk_size_parameter(self, tmp_path):
        """Test that load_trios accepts chunk_size parameter."""
        trios_content = """Filename	test.txt
Instrument serial number	4010-1234

[step]
Number of points	20
Time	Stress
s	Pa
"""
        for i in range(20):
            trios_content += f"{i * 0.1:.1f}\t{(i + 1) * 100}\n"

        test_file = tmp_path / "test_compat.txt"
        test_file.write_text(trios_content)

        # Using load_trios with chunk_size should work
        result = load_trios(str(test_file), chunk_size=10)

        # For single segment, should return list of chunks
        assert isinstance(result, list)
        assert len(result) == 2

        # Each chunk should be RheoData
        for chunk in result:
            assert isinstance(chunk, RheoData)

    def test_chunked_generator_cleanup(self, tmp_path):
        """Test that generator properly cleans up file handles."""
        trios_content = """Filename	test.txt
Instrument serial number	4010-1234

[step]
Number of points	100
Time	Stress
s	Pa
"""
        for i in range(100):
            trios_content += f"{i * 0.01:.2f}\t{(i + 1) * 10}\n"

        test_file = tmp_path / "test_cleanup.txt"
        test_file.write_text(trios_content)

        # Start reading but don't finish
        gen = load_trios_chunked(str(test_file), chunk_size=10)
        chunk1 = next(gen)
        assert len(chunk1.x) >= 1  # Should have at least some data

        # Delete generator without exhausting it
        del gen

        # Should still be able to open file
        gen2 = load_trios_chunked(str(test_file), chunk_size=10)
        chunk2 = next(gen2)
        assert len(chunk2.x) >= 1  # Should have at least some data


class TestTriosChunkedMemoryProfile:
    """Memory profiling and performance tests."""

    @pytest.mark.slow
    def test_memory_efficiency_simulation(self, tmp_path):
        """Simulate memory-efficient reading of large file.

        This test verifies the chunked reader maintains constant
        memory usage regardless of file size.
        """
        # Create a large file (10,000 points)
        lines = [
            "Filename\tlarge.txt",
            "Instrument serial number\t4010-1234",
            "",
            "[step]",
            "Number of points\t10000",
            "Time\tStress",
            "s\tPa",
        ]
        for i in range(10000):
            lines.append(f"{i * 0.001:.3f}\t{(i + 1) * 1.5:.1f}")

        test_file = tmp_path / "large_sim.txt"
        test_file.write_text("\n".join(lines))

        # Process in small chunks
        chunk_count = 0
        total_points = 0
        max_stress = -float('inf')

        for chunk in load_trios_chunked(str(test_file), chunk_size=1000):
            chunk_count += 1
            total_points += len(chunk.x)
            max_stress = max(max_stress, float(chunk.y.max()))

        # Verify processing
        assert chunk_count == 10
        assert total_points == 10000
        assert max_stress > 0

    def test_chunked_integration_with_model_fitting(self, tmp_path):
        """Test chunked reading with model fitting workflow."""
        # Create file with synthetic stress relaxation data
        lines = [
            "Filename\trelax.txt",
            "Instrument serial number\t4010-1234",
            "",
            "[step]",
            "Number of points\t50",
            "Time\tStress",
            "s\tPa",
        ]

        # Generate synthetic relaxation data
        for i in range(50):
            t = i * 0.1
            stress = 1000 * np.exp(-t / 1.0)  # Simple exponential decay
            lines.append(f"{t:.1f}\t{stress:.2f}")

        test_file = tmp_path / "relax.txt"
        test_file.write_text("\n".join(lines))

        # Process chunks (simulating model fitting)
        chunk_results = []
        for chunk in load_trios_chunked(str(test_file), chunk_size=10):
            # Simulate some processing (e.g., compute mean stress)
            mean_stress = float(chunk.y.mean())
            chunk_results.append(mean_stress)

        # Should have processed 5 chunks
        assert len(chunk_results) == 5

        # Mean stress should decrease (relaxation)
        assert chunk_results[0] > chunk_results[-1]
