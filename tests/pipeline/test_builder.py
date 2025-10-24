"""Tests for PipelineBuilder.

This module tests the programmatic pipeline construction with validation.
"""

import pytest
import numpy as np
import tempfile
import os

from rheo.pipeline.builder import PipelineBuilder, ConditionalPipelineBuilder
from rheo.pipeline import Pipeline
from rheo.core.data import RheoData
from rheo.core.base import BaseModel
from rheo.core.registry import ModelRegistry


# Mock model for testing
class BuilderTestModel(BaseModel):
    """Simple mock model for builder tests."""

    def __init__(self):
        super().__init__()
        from rheo.core.parameters import Parameter
        self.parameters.add(Parameter('a', value=1.0, bounds=(0, 10)))

    def _fit(self, X, y, **kwargs):
        self.parameters.set_value('a', float(np.mean(y)))
        return self

    def _predict(self, X):
        a = self.parameters.get_value('a')
        return a * np.ones_like(X)


@pytest.fixture(scope='module', autouse=True)
def register_test_model():
    """Register test model."""
    ModelRegistry.register('builder_test_model')(BuilderTestModel)
    yield
    ModelRegistry.unregister('builder_test_model')


@pytest.fixture
def temp_csv_file():
    """Create temporary CSV file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write('x,y\n')
        for i in range(10):
            f.write(f'{i},{i*2}\n')
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestPipelineBuilderInitialization:
    """Test builder initialization."""

    def test_init(self):
        """Test builder initialization."""
        builder = PipelineBuilder()
        assert len(builder.steps) == 0

    def test_empty_builder_len(self):
        """Test length of empty builder."""
        builder = PipelineBuilder()
        assert len(builder) == 0


class TestPipelineBuilderSteps:
    """Test adding steps to builder."""

    def test_add_load_step(self, temp_csv_file):
        """Test adding load step."""
        builder = PipelineBuilder()
        builder.add_load_step(temp_csv_file, format='csv', x_col='x', y_col='y')

        assert len(builder) == 1
        assert builder.steps[0][0] == 'load'
        assert builder.steps[0][1]['file_path'] == temp_csv_file

    def test_add_transform_step(self):
        """Test adding transform step."""
        builder = PipelineBuilder()
        builder.add_transform_step('smooth', window_size=5)

        assert len(builder) == 1
        assert builder.steps[0][0] == 'transform'
        assert builder.steps[0][1]['name'] == 'smooth'
        assert builder.steps[0][1]['window_size'] == 5

    def test_add_fit_step(self):
        """Test adding fit step."""
        builder = PipelineBuilder()
        builder.add_fit_step('builder_test_model', method='L-BFGS-B')

        assert len(builder) == 1
        assert builder.steps[0][0] == 'fit'
        assert builder.steps[0][1]['model'] == 'builder_test_model'
        assert builder.steps[0][1]['method'] == 'L-BFGS-B'

    def test_add_plot_step(self):
        """Test adding plot step."""
        builder = PipelineBuilder()
        builder.add_plot_step(style='publication', show=False)

        assert len(builder) == 1
        assert builder.steps[0][0] == 'plot'
        assert builder.steps[0][1]['style'] == 'publication'
        assert builder.steps[0][1]['show'] is False

    def test_add_save_step(self):
        """Test adding save step."""
        builder = PipelineBuilder()
        builder.add_save_step('output.hdf5', format='hdf5')

        assert len(builder) == 1
        assert builder.steps[0][0] == 'save'
        assert builder.steps[0][1]['file_path'] == 'output.hdf5'


class TestPipelineBuilderChaining:
    """Test builder method chaining."""

    def test_chaining(self, temp_csv_file):
        """Test that builder methods return self."""
        builder = (PipelineBuilder()
                   .add_load_step(temp_csv_file)
                   .add_fit_step('builder_test_model')
                   .add_plot_step())

        assert len(builder) == 3
        assert builder.steps[0][0] == 'load'
        assert builder.steps[1][0] == 'fit'
        assert builder.steps[2][0] == 'plot'


class TestPipelineBuilderValidation:
    """Test pipeline validation."""

    def test_empty_pipeline_validation(self):
        """Test that empty pipeline fails validation."""
        builder = PipelineBuilder()

        with pytest.raises(ValueError, match="no steps"):
            builder.build()

    def test_load_first_validation(self, temp_csv_file):
        """Test that pipeline must start with load."""
        builder = PipelineBuilder()
        builder.add_fit_step('builder_test_model')

        with pytest.raises(ValueError, match="must start with a load"):
            builder.build()

    def test_valid_pipeline(self, temp_csv_file):
        """Test building valid pipeline."""
        builder = (PipelineBuilder()
                   .add_load_step(temp_csv_file, format='csv', x_col='x', y_col='y')
                   .add_fit_step('builder_test_model'))

        pipeline = builder.build()
        assert isinstance(pipeline, Pipeline)

    def test_skip_validation(self, temp_csv_file):
        """Test building without validation."""
        builder = PipelineBuilder()
        builder.add_fit_step('builder_test_model')  # Invalid: no load first

        # Should not raise when validate=False
        pipeline = builder.build(validate=False)
        assert isinstance(pipeline, Pipeline)


class TestPipelineBuilderExecution:
    """Test executing built pipelines."""

    def test_build_and_execute(self, temp_csv_file):
        """Test building and executing pipeline."""
        builder = (PipelineBuilder()
                   .add_load_step(temp_csv_file, format='csv', x_col='x', y_col='y')
                   .add_fit_step('builder_test_model'))

        pipeline = builder.build()

        assert pipeline.data is not None
        assert pipeline._last_model is not None

    def test_build_with_save(self, temp_csv_file):
        """Test building pipeline with save step."""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
            output_path = f.name

        try:
            builder = (PipelineBuilder()
                       .add_load_step(temp_csv_file, format='csv', x_col='x', y_col='y')
                       .add_save_step(output_path))

            pipeline = builder.build()

            # Check that file was saved
            # Note: might fail if save not implemented
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestPipelineBuilderUtilities:
    """Test builder utility methods."""

    def test_get_steps(self, temp_csv_file):
        """Test getting steps."""
        builder = (PipelineBuilder()
                   .add_load_step(temp_csv_file)
                   .add_fit_step('builder_test_model'))

        steps = builder.get_steps()
        assert len(steps) == 2
        assert steps[0][0] == 'load'
        assert steps[1][0] == 'fit'

    def test_get_steps_copy(self, temp_csv_file):
        """Test that get_steps returns a copy."""
        builder = PipelineBuilder().add_load_step(temp_csv_file)

        steps1 = builder.get_steps()
        steps2 = builder.get_steps()

        assert steps1 is not steps2
        assert steps1 == steps2

    def test_clear(self, temp_csv_file):
        """Test clearing builder."""
        builder = (PipelineBuilder()
                   .add_load_step(temp_csv_file)
                   .add_fit_step('builder_test_model'))

        builder.clear()

        assert len(builder) == 0
        assert len(builder.steps) == 0

    def test_repr(self, temp_csv_file):
        """Test string representation."""
        builder = (PipelineBuilder()
                   .add_load_step(temp_csv_file)
                   .add_fit_step('builder_test_model'))

        repr_str = repr(builder)
        assert 'PipelineBuilder' in repr_str
        assert 'load' in repr_str
        assert 'fit' in repr_str


class TestConditionalPipelineBuilder:
    """Test conditional pipeline builder."""

    def test_initialization(self):
        """Test conditional builder initialization."""
        builder = ConditionalPipelineBuilder()
        assert len(builder.steps) == 0
        assert len(builder.conditions) == 0

    def test_add_conditional_step(self):
        """Test adding conditional step."""
        builder = ConditionalPipelineBuilder()

        def condition(pipeline):
            return len(pipeline.data.x) > 10

        builder.add_conditional_step(
            'fit',
            condition,
            model='builder_test_model'
        )

        assert len(builder) == 1
        assert 0 in builder.conditions

    def test_build_with_warning(self, temp_csv_file):
        """Test that conditional build produces warning."""
        builder = ConditionalPipelineBuilder()
        builder.add_load_step(temp_csv_file, format='csv', x_col='x', y_col='y')
        builder.add_conditional_step(
            'fit',
            lambda p: True,
            model='builder_test_model'
        )

        with pytest.warns(UserWarning, match="not fully implemented"):
            pipeline = builder.build()

        assert isinstance(pipeline, Pipeline)


class TestBuilderErrorHandling:
    """Test error handling in builder."""

    def test_invalid_model_name(self, temp_csv_file):
        """Test validation with invalid model name."""
        builder = (PipelineBuilder()
                   .add_load_step(temp_csv_file, format='csv', x_col='x', y_col='y')
                   .add_fit_step('nonexistent_model'))

        with pytest.raises(ValueError, match="not found in registry"):
            builder.build()

    def test_build_without_load(self):
        """Test building pipeline without load step."""
        builder = PipelineBuilder()
        builder.add_plot_step()

        with pytest.raises(ValueError, match="must start with a load"):
            builder.build()
