"""Pytest configuration and fixtures for inspect_mlflow tests."""

import os
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def clean_env(monkeypatch):
    """Remove all INSPECT_MLFLOW_ environment variables."""
    env_vars = [k for k in os.environ if k.startswith("INSPECT_MLFLOW_")]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)
    yield


@pytest.fixture
def mock_mlflow():
    """Create a mock MLflow module."""
    mock = MagicMock()
    mock.active_run.return_value = None
    mock.start_run.return_value = MagicMock(info=MagicMock(run_id="test-run-123"))
    mock.tracking.MlflowClient.return_value = MagicMock(
        get_experiment_by_name=MagicMock(return_value=None),
        create_experiment=MagicMock(return_value="exp-123"),
    )
    return mock


@pytest.fixture
def sample_task_spec():
    """Create a mock TaskStart spec."""
    spec = MagicMock()
    spec.task = "test_task"
    spec.name = "test_task"
    spec.model = "openai/gpt-4"
    spec.solver = "chain_of_thought"
    spec.metadata = None
    spec.dataset = MagicMock(name="test_dataset", size=10)
    spec.task_file = "test.py"
    spec.task_version = 1
    spec.task_id = "task-123"
    return spec


@pytest.fixture
def sample_eval_sample():
    """Create a mock EvalSample."""
    sample = MagicMock()
    sample.id = "sample-1"
    sample.input = "What is 2+2?"
    sample.target = "4"
    sample.output = MagicMock(completion="The answer is 4")
    sample.scores = {"accuracy": MagicMock(value="C", name="accuracy")}
    sample.events = []
    sample.model_usage = {
        "openai/gpt-4": MagicMock(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )
    }
    sample.total_time = 1.5
    sample.working_time = 1.0
    sample.error = None
    return sample
