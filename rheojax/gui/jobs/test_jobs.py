"""
Test script for background job system.

This demonstrates the functionality without requiring PySide6 GUI.
"""

import numpy as np

from rheojax.gui.jobs import CancellationError, CancellationToken


def test_cancellation_token():
    """Test CancellationToken functionality."""
    print("Testing CancellationToken...")

    # Create token
    token = CancellationToken()
    assert not token.is_cancelled(), "Token should not be cancelled initially"

    # Cancel token
    token.cancel()
    assert token.is_cancelled(), "Token should be cancelled"

    # Check should raise
    try:
        token.check()
        raise AssertionError("check() should raise CancellationError")
    except CancellationError:
        pass  # Expected

    # Test error storage
    test_error = ValueError("Test error")
    token.set_error(test_error)
    retrieved_error = token.get_error()
    assert retrieved_error is test_error, "Should retrieve stored error"

    # Reset token
    token.reset()
    assert not token.is_cancelled(), "Token should be reset"
    assert token.get_error() is None, "Error should be cleared"

    print("[OK] CancellationToken tests passed")


def test_cancellation_workflow():
    """Test cancellation in a simulated workflow."""
    print("\nTesting cancellation workflow...")

    token = CancellationToken()
    iterations = 0
    max_iterations = 100

    try:
        for i in range(max_iterations):
            # Simulate work
            iterations += 1

            # Cancel after 10 iterations
            if i == 10:
                token.cancel()

            # Check for cancellation
            token.check()

        raise AssertionError("Should have been cancelled")

    except CancellationError:
        assert iterations == 11, f"Should have done 11 iterations, got {iterations}"

    print(f"[OK] Cancelled after {iterations} iterations")


def test_error_handling():
    """Test error storage and retrieval."""
    print("\nTesting error handling...")

    token = CancellationToken()

    # Store multiple errors (last one wins)
    error1 = ValueError("First error")
    error2 = RuntimeError("Second error")

    token.set_error(error1)
    assert token.get_error() is error1

    token.set_error(error2)
    assert token.get_error() is error2

    # Reset clears error
    token.reset()
    assert token.get_error() is None

    print("[OK] Error handling tests passed")


def test_wait_timeout():
    """Test wait with timeout."""
    print("\nTesting wait timeout...")
    import time

    token = CancellationToken()

    # Test timeout
    start = time.time()
    result = token.wait(timeout=0.1)
    elapsed = time.time() - start

    assert not result, "Should timeout without cancellation"
    assert 0.09 < elapsed < 0.2, f"Timeout duration incorrect: {elapsed}"

    # Test immediate return when cancelled
    token.cancel()
    start = time.time()
    result = token.wait(timeout=1.0)
    elapsed = time.time() - start

    assert result, "Should return immediately when cancelled"
    assert elapsed < 0.1, f"Should return immediately, got {elapsed}"

    print("[OK] Wait timeout tests passed")


def test_fit_result_structure():
    """Test FitResult dataclass."""
    print("\nTesting FitResult structure...")
    from datetime import datetime

    from rheojax.gui.jobs import FitResult

    result = FitResult(
        model_name="maxwell",
        parameters={"G0": 1e6, "tau": 1.0},
        r_squared=0.95,
        mpe=2.5,
        chi_squared=0.01,
        fit_time=1.5,
        timestamp=datetime.now(),
        num_iterations=100,
        success=True,
    )

    assert result.model_name == "maxwell"
    assert result.parameters["G0"] == 1e6
    assert result.r_squared == 0.95
    assert result.success is True

    print("[OK] FitResult structure tests passed")


def test_bayesian_result_structure():
    """Test BayesianResult dataclass."""
    print("\nTesting BayesianResult structure...")
    from datetime import datetime

    from rheojax.gui.jobs import BayesianResult

    posterior_samples = {
        "G0": np.random.randn(2000),
        "tau": np.random.randn(2000),
    }

    summary = {
        "G0": {"mean": 1e6, "std": 1e5},
        "tau": {"mean": 1.0, "std": 0.1},
    }

    result = BayesianResult(
        model_name="maxwell",
        dataset_id="test-ds-001",
        posterior_samples=posterior_samples,
        summary=summary,
        r_hat={"G0": 1.001, "tau": 1.002},
        ess={"G0": 500.0, "tau": 450.0},
        divergences=0,
        credible_intervals={"G0": (8e5, 1.2e6), "tau": (0.8, 1.2)},
        mcmc_time=10.5,
        timestamp=datetime.now(),
        num_samples=2000,
        num_chains=4,
        num_warmup=1000,
    )

    assert result.model_name == "maxwell"
    assert result.dataset_id == "test-ds-001"
    assert result.num_samples == 2000
    assert result.num_chains == 4
    assert result.num_warmup == 1000
    # diagnostics is now a computed property returning a dict
    assert result.diagnostics["divergences"] == 0
    assert result.diagnostics["r_hat"]["G0"] == 1.001
    assert result.diagnostics["ess"]["tau"] == 450.0
    # sampling_time is a property alias for mcmc_time
    assert result.sampling_time == 10.5
    assert result.mcmc_time == 10.5
    # metadata is a synthesized property
    assert result.metadata["model_name"] == "maxwell"
    assert result.metadata["num_chains"] == 4

    print("[OK] BayesianResult structure tests passed")


if __name__ == "__main__":
    print("=" * 60)
    print("RheoJAX Background Job System Tests")
    print("=" * 60)

    test_cancellation_token()
    test_cancellation_workflow()
    test_error_handling()
    test_wait_timeout()
    test_fit_result_structure()
    test_bayesian_result_structure()

    print("\n" + "=" * 60)
    print("All tests passed! [OK]")
    print("=" * 60)
