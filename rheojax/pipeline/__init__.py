"""Pipeline API for rheological analysis workflows.

This module provides a fluent API for chaining rheological analysis operations,
enabling intuitive workflows from data loading through modeling and visualization.

Main Components:
    - Pipeline: Core fluent API for method chaining
    - BayesianPipeline: Specialized pipeline for NLSQ → NUTS Bayesian workflows
    - Workflow pipelines: Pre-configured pipelines for common tasks
    - PipelineBuilder: Programmatic pipeline construction
    - BatchPipeline: Process multiple datasets

Example - Basic workflow:
    >>> from rheojax.pipeline import Pipeline
    >>> pipeline = (Pipeline()
    ...     .load('data.csv')
    ...     .fit('maxwell')
    ...     .plot()
    ...     .save('result.hdf5'))

Example - Bayesian workflow:
    >>> from rheojax.pipeline import BayesianPipeline
    >>> pipeline = (BayesianPipeline()
    ...     .load('data.csv')
    ...     .fit_nlsq('maxwell')
    ...     .fit_bayesian(num_samples=2000)
    ...     .plot_posterior()
    ...     .save('results.hdf5'))

Example - Model comparison:
    >>> from rheojax.pipeline import ModelComparisonPipeline
    >>> pipeline = ModelComparisonPipeline(['maxwell', 'zener', 'springpot'])
    >>> pipeline.run(data)
    >>> best = pipeline.get_best_model()

Example - Batch processing:
    >>> from rheojax.pipeline import Pipeline, BatchPipeline
    >>> template = Pipeline().fit('maxwell')
    >>> batch = BatchPipeline(template)
    >>> batch.process_directory('data/')
    >>> batch.export_summary('summary.xlsx')

Example - Pipeline builder:
    >>> from rheojax.pipeline import PipelineBuilder
    >>> pipeline = (PipelineBuilder()
    ...     .add_load_step('data.csv')
    ...     .add_transform_step('smooth', window_size=5)
    ...     .add_fit_step('maxwell')
    ...     .build())
"""

from rheojax.pipeline.base import Pipeline
from rheojax.pipeline.batch import BatchPipeline
from rheojax.pipeline.bayesian import BayesianPipeline
from rheojax.pipeline.builder import (
    ConditionalPipelineBuilder,
    PipelineBuilder,
)
from rheojax.pipeline.workflows import (
    CreepToRelaxationPipeline,
    FrequencyToTimePipeline,
    MastercurvePipeline,
    ModelComparisonPipeline,
    SPPAmplitudeSweepPipeline,
)

__all__ = [
    # Core pipeline
    "Pipeline",
    "BayesianPipeline",
    # Workflow pipelines
    "MastercurvePipeline",
    "ModelComparisonPipeline",
    "CreepToRelaxationPipeline",
    "FrequencyToTimePipeline",
    "SPPAmplitudeSweepPipeline",
    # Builders
    "PipelineBuilder",
    "ConditionalPipelineBuilder",
    # Batch processing
    "BatchPipeline",
]
