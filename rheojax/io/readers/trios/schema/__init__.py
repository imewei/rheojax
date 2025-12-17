"""TRIOS JSON schema dataclasses.

This module provides dataclasses for parsing TRIOS JSON export files
based on the official TA Instruments TRIOS JSON Export Schema.

Usage:
    >>> from rheojax.io.readers.trios.schema import TRIOSExperiment
    >>> import json
    >>> with open('data.json') as f:
    ...     data = json.load(f)
    >>> experiment = TRIOSExperiment.from_json(data)
    >>> df = experiment.get_dataframe()
"""

from rheojax.io.readers.trios.schema.dataset import TRIOSDataSet
from rheojax.io.readers.trios.schema.experiment import TRIOSExperiment, TRIOSResult

__all__ = [
    "TRIOSDataSet",
    "TRIOSExperiment",
    "TRIOSResult",
]
