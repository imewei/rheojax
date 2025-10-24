"""Pipeline API for rheological analysis workflows.

This module provides a fluent API for chaining rheological analysis operations,
enabling intuitive workflows from data loading through modeling and visualization.

Main Components:
    - Pipeline: Core fluent API for method chaining
    - Workflow pipelines: Pre-configured pipelines for common tasks
    - PipelineBuilder: Programmatic pipeline construction
    - BatchPipeline: Process multiple datasets

Example - Basic workflow:
    >>> from rheo.pipeline import Pipeline
    >>> pipeline = (Pipeline()
    ...     .load('data.csv')
    ...     .fit('maxwell')
    ...     .plot()
    ...     .save('result.hdf5'))

Example - Model comparison:
    >>> from rheo.pipeline import ModelComparisonPipeline
    >>> pipeline = ModelComparisonPipeline(['maxwell', 'zener', 'springpot'])
    >>> pipeline.run(data)
    >>> best = pipeline.get_best_model()

Example - Batch processing:
    >>> from rheo.pipeline import Pipeline, BatchPipeline
    >>> template = Pipeline().fit('maxwell')
    >>> batch = BatchPipeline(template)
    >>> batch.process_directory('data/')
    >>> batch.export_summary('summary.xlsx')

Example - Pipeline builder:
    >>> from rheo.pipeline import PipelineBuilder
    >>> pipeline = (PipelineBuilder()
    ...     .add_load_step('data.csv')
    ...     .add_transform_step('smooth', window_size=5)
    ...     .add_fit_step('maxwell')
    ...     .build())
"""

from rheo.pipeline.base import Pipeline
from rheo.pipeline.workflows import (
    MastercurvePipeline,
    ModelComparisonPipeline,
    CreepToRelaxationPipeline,
    FrequencyToTimePipeline,
)
from rheo.pipeline.builder import (
    PipelineBuilder,
    ConditionalPipelineBuilder,
)
from rheo.pipeline.batch import BatchPipeline

__all__ = [
    # Core pipeline
    'Pipeline',

    # Workflow pipelines
    'MastercurvePipeline',
    'ModelComparisonPipeline',
    'CreepToRelaxationPipeline',
    'FrequencyToTimePipeline',

    # Builders
    'PipelineBuilder',
    'ConditionalPipelineBuilder',

    # Batch processing
    'BatchPipeline',
]
