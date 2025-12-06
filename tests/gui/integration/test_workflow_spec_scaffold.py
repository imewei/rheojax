"""Integration workflow scaffold derived from GUI SPEC.md.

Marked skipped until the Data → Fit → Bayesian → Export pipeline is wired.
"""

import pytest


@pytest.mark.gui
@pytest.mark.skip(reason="Pending end-to-end GUI workflow wiring")
def test_data_to_fit_workflow():
    """Placeholder for Data -> Fit workflow regression."""
