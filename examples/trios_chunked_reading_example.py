"""Usage examples for TRIOS chunked reading.

This script demonstrates memory-efficient chunked reading of large TRIOS files,
particularly useful for OWChirp arbitrary wave files (66-80 MB, 150k+ points).

Memory Savings:
- Traditional loading: ~12 MB for 150k points
- Chunked loading (10k chunks): ~800 KB peak memory usage
- Reduction: ~93% memory savings

Author: Rheo development team
Date: 2025-10-31
"""

from pathlib import Path

import numpy as np

from rheojax.io.readers.trios import load_trios, load_trios_chunked
from rheojax.models.maxwell import Maxwell

# =============================================================================
# Example 1: Basic Chunked Reading
# =============================================================================


def example_basic_chunked_reading():
    """Basic chunked reading example - iterate through chunks."""
    print("=" * 70)
    print("Example 1: Basic Chunked Reading")
    print("=" * 70)

    # For large files (> 10 MB, > 50k points), use chunked reading
    for i, chunk in enumerate(load_trios_chunked('large_file.txt', chunk_size=10000)):
        print(f"Chunk {i + 1}:")
        print(f"  Points: {len(chunk.x)}")
        print(f"  Time range: {chunk.x.min():.3f} - {chunk.x.max():.3f} s")
        print(f"  Stress range: {chunk.y.min():.2f} - {chunk.y.max():.2f} Pa")
        print()

    # Notes:
    # - Each chunk is a complete RheoData object with metadata
    # - Chunk boundaries are arbitrary (based on row count, not time)
    # - File handle automatically closes when iteration completes


# =============================================================================
# Example 2: Aggregating Results Across Chunks
# =============================================================================


def example_aggregate_statistics():
    """Compute statistics across chunks without loading entire file."""
    print("=" * 70)
    print("Example 2: Aggregating Statistics")
    print("=" * 70)

    # Compute global statistics efficiently
    total_points = 0
    max_stress = -float('inf')
    min_stress = float('inf')
    sum_stress = 0.0

    for chunk in load_trios_chunked('large_file.txt', chunk_size=10000):
        total_points += len(chunk.x)
        max_stress = max(max_stress, float(chunk.y.max()))
        min_stress = min(min_stress, float(chunk.y.min()))
        sum_stress += float(chunk.y.sum())

    mean_stress = sum_stress / total_points

    print(f"Total data points: {total_points}")
    print(f"Max stress: {max_stress:.2f} Pa")
    print(f"Min stress: {min_stress:.2f} Pa")
    print(f"Mean stress: {mean_stress:.2f} Pa")
    print()

    # Memory usage: Only ~800 KB at any time (for 10k chunk_size)
    # vs ~12 MB for full file load


# =============================================================================
# Example 3: Model Fitting on Chunks
# =============================================================================


def example_model_fitting_chunks():
    """Fit model to each chunk independently."""
    print("=" * 70)
    print("Example 3: Model Fitting on Chunks")
    print("=" * 70)

    model = Maxwell()
    chunk_results = []

    for i, chunk in enumerate(load_trios_chunked('relaxation_data.txt', chunk_size=5000)):
        # Fit Maxwell model to this chunk
        model.fit(chunk.x, chunk.y)

        # Store results
        G0 = model.parameters.get_value('G0')
        eta = model.parameters.get_value('eta')

        chunk_results.append({
            'chunk': i + 1,
            'points': len(chunk.x),
            'G0': G0,
            'eta': eta,
            'time_range': (chunk.x.min(), chunk.x.max())
        })

        print(f"Chunk {i + 1}: G0 = {G0:.3e} Pa, eta = {eta:.3e} Pa·s")

    # Analyze how parameters evolve across chunks
    print("\nParameter evolution:")
    print(f"  G0 variation: {np.std([r['G0'] for r in chunk_results]):.3e} Pa")
    print(f"  eta variation: {np.std([r['eta'] for r in chunk_results]):.3e} Pa·s")
    print()


# =============================================================================
# Example 4: Processing Specific Segment
# =============================================================================


def example_segment_selection():
    """Process only a specific segment from multi-segment file."""
    print("=" * 70)
    print("Example 4: Segment Selection")
    print("=" * 70)

    # TRIOS files often contain multiple procedure steps
    # Process only segment 2 (0-indexed)

    print("Processing only segment index 1:")
    for i, chunk in enumerate(
        load_trios_chunked(
            'multi_step.txt',
            chunk_size=10000,
            segment_index=1  # Second segment
        )
    ):
        print(f"  Chunk {i + 1}: {len(chunk.x)} points")
        print(f"    Test mode: {chunk.metadata.get('test_mode', 'unknown')}")

    print()


# =============================================================================
# Example 5: Comparison - Chunked vs Full Load
# =============================================================================


def example_compare_methods(filepath: str):
    """Compare chunked vs full loading for same file."""
    print("=" * 70)
    print("Example 5: Chunked vs Full Loading Comparison")
    print("=" * 70)

    # Method 1: Full load (high memory)
    print("Method 1: Full loading")
    full_data = load_trios(filepath)
    print(f"  Total points: {len(full_data.x)}")
    print(f"  Memory usage: ~{len(full_data.x) * 80 / 1024 / 1024:.1f} MB")
    print()

    # Method 2: Chunked load (low memory)
    print("Method 2: Chunked loading (10k chunks)")
    chunk_count = 0
    total_points_chunked = 0

    for chunk in load_trios_chunked(filepath, chunk_size=10000):
        chunk_count += 1
        total_points_chunked += len(chunk.x)

    print(f"  Total chunks: {chunk_count}")
    print(f"  Total points: {total_points_chunked}")
    print(f"  Peak memory usage: ~{10000 * 80 / 1024 / 1024:.1f} MB")
    print()

    # Verify equivalence
    assert len(full_data.x) == total_points_chunked
    print("✓ Both methods processed same number of points")
    print()


# =============================================================================
# Example 6: Backward Compatibility - Using load_trios with chunk_size
# =============================================================================


def example_backward_compatibility():
    """Use load_trios with chunk_size parameter for compatibility."""
    print("=" * 70)
    print("Example 6: Backward Compatibility")
    print("=" * 70)

    # Option 1: Direct chunked reading (recommended)
    chunks_list = list(load_trios_chunked('data.txt', chunk_size=10000))
    print(f"Option 1 (load_trios_chunked): {len(chunks_list)} chunks")

    # Option 2: Using load_trios with chunk_size (backward compatible)
    result = load_trios('data.txt', chunk_size=10000)
    if isinstance(result, list):
        print(f"Option 2 (load_trios): {len(result)} chunks")
    else:
        print(f"Option 2 (load_trios): Single RheoData object")

    print("\nBoth methods produce identical results.")
    print()


# =============================================================================
# Example 7: Real-World OWChirp Processing
# =============================================================================


def example_owchirp_processing():
    """Process large OWChirp arbitrary wave file efficiently.

    OWChirp files are typically:
    - 66-80 MB file size
    - 150,000+ data points
    - High-frequency sampling
    """
    print("=" * 70)
    print("Example 7: OWChirp File Processing")
    print("=" * 70)

    owchirp_file = 'owchirp_data.txt'

    print(f"Processing OWChirp file: {owchirp_file}")
    print("File characteristics:")
    print("  - Size: ~70 MB")
    print("  - Points: ~150,000")
    print("  - Traditional loading: ~12 MB memory")
    print("  - Chunked loading: ~800 KB memory (10k chunks)")
    print()

    # Process in chunks
    print("Processing chunks:")
    chunk_statistics = []

    for i, chunk in enumerate(load_trios_chunked(owchirp_file, chunk_size=10000)):
        # Compute chunk statistics
        stats = {
            'chunk': i + 1,
            'points': len(chunk.x),
            'time_start': float(chunk.x[0]),
            'time_end': float(chunk.x[-1]),
            'stress_mean': float(chunk.y.mean()),
            'stress_std': float(chunk.y.std()),
            'stress_max': float(chunk.y.max()),
        }
        chunk_statistics.append(stats)

        # Print progress every 5 chunks
        if (i + 1) % 5 == 0:
            print(f"  Processed {i + 1} chunks ({(i + 1) * 10000} points)...")

    print(f"\n✓ Completed: {len(chunk_statistics)} chunks processed")
    print(f"  Total points: {sum(s['points'] for s in chunk_statistics)}")
    print(f"  Global max stress: {max(s['stress_max'] for s in chunk_statistics):.2f} Pa")
    print()


# =============================================================================
# Example 8: Validation Disabled for Speed
# =============================================================================


def example_validation_disabled():
    """Disable validation for maximum reading speed."""
    print("=" * 70)
    print("Example 8: Validation Disabled")
    print("=" * 70)

    print("Reading with validation disabled (faster, but less safe):")

    chunk_count = 0
    for chunk in load_trios_chunked(
        'large_file.txt',
        chunk_size=10000,
        validate_data=False  # Skip validation for speed
    ):
        chunk_count += 1

    print(f"  Processed {chunk_count} chunks")
    print("\nNote: Validation checks for NaN, non-finite values, and monotonicity.")
    print("      Disable only if you trust the data quality.")
    print()


# =============================================================================
# Example 9: Memory-Efficient Data Export
# =============================================================================


def example_chunked_export():
    """Export data in chunks to avoid memory issues."""
    print("=" * 70)
    print("Example 9: Chunked Data Export")
    print("=" * 70)

    output_file = Path('processed_data.csv')

    print(f"Exporting processed data to: {output_file}")

    # Write header
    with open(output_file, 'w') as f:
        f.write("time,stress,processing_flag\n")

    # Process and write chunks
    chunk_count = 0
    for chunk in load_trios_chunked('large_file.txt', chunk_size=10000):
        # Process chunk (e.g., apply threshold)
        threshold = 1000.0
        flags = chunk.y > threshold

        # Append to file
        with open(output_file, 'a') as f:
            for t, s, flag in zip(chunk.x, chunk.y, flags):
                f.write(f"{t:.6f},{s:.2f},{int(flag)}\n")

        chunk_count += 1

    print(f"✓ Exported {chunk_count} chunks")
    print(f"  Output file: {output_file}")
    print()


# =============================================================================
# Example 10: Choosing Optimal Chunk Size
# =============================================================================


def example_chunk_size_selection():
    """Guide for selecting optimal chunk size."""
    print("=" * 70)
    print("Example 10: Choosing Optimal Chunk Size")
    print("=" * 70)

    file_size_mb = 70  # Example: 70 MB file
    estimated_points = 150000  # Example: 150k points

    print("Chunk Size Guidelines:")
    print()

    # Small chunks (1,000 - 5,000)
    print("Small chunks (1,000 - 5,000 points):")
    print("  Memory: ~80-400 KB")
    print("  Use case: Very limited memory, point-by-point processing")
    print("  Trade-off: More overhead, slower overall")
    print()

    # Medium chunks (5,000 - 20,000) - RECOMMENDED
    print("Medium chunks (5,000 - 20,000 points): [RECOMMENDED]")
    print("  Memory: ~400 KB - 1.6 MB")
    print("  Use case: Most applications, good balance")
    print("  Trade-off: Optimal for typical workflows")
    print()

    # Large chunks (20,000 - 50,000)
    print("Large chunks (20,000 - 50,000 points):")
    print("  Memory: ~1.6 - 4 MB")
    print("  Use case: Ample memory, fewer iterations needed")
    print("  Trade-off: Higher memory, faster processing")
    print()

    # Recommendation for this file
    recommended_chunk_size = 10000
    estimated_chunks = (estimated_points + recommended_chunk_size - 1) // recommended_chunk_size
    memory_per_chunk_mb = recommended_chunk_size * 80 / 1024 / 1024

    print(f"For your file ({file_size_mb} MB, ~{estimated_points} points):")
    print(f"  Recommended chunk size: {recommended_chunk_size}")
    print(f"  Number of chunks: ~{estimated_chunks}")
    print(f"  Memory per chunk: ~{memory_per_chunk_mb:.1f} MB")
    print(f"  Total memory saved: ~{file_size_mb - memory_per_chunk_mb:.1f} MB")
    print()


# =============================================================================
# Main - Run all examples
# =============================================================================


def main():
    """Run all examples (if files exist)."""
    print("\n" + "=" * 70)
    print("TRIOS CHUNKED READING EXAMPLES")
    print("=" * 70)
    print()
    print("This script demonstrates memory-efficient chunked reading.")
    print("Note: Examples require actual TRIOS files to run.")
    print()

    # Example function calls would go here
    # Commented out since files may not exist

    # example_basic_chunked_reading()
    # example_aggregate_statistics()
    # example_model_fitting_chunks()
    # example_segment_selection()
    # example_backward_compatibility()
    # example_owchirp_processing()
    # example_validation_disabled()
    # example_chunked_export()

    # Always run - doesn't need files
    example_chunk_size_selection()

    print("=" * 70)
    print("For more information:")
    print("  - Documentation: rheo.io.readers.trios.load_trios_chunked")
    print("  - Tests: tests/io/test_trios_chunked.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
