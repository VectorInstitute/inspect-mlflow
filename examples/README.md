# Examples

These examples are for quickly validating that `inspect-mlflow` logs Inspect runs correctly into MLflow.
Below are the steps to run them locally.

## Setup server

1. Start MLflow UI:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

## Run locally

After setup, configure `INSPECT_MLFLOW_TRACKING_URI` in one of these two ways.

1. Add it to a `.env` file used by Inspect:

```bash
INSPECT_MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

2. Or export it in your terminal:

```bash
export INSPECT_MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Then run any of the included evals:

```bash
uv run inspect eval examples/hello_world.py
uv run inspect eval examples/parallel_add.py
uv run inspect eval examples/surfer.py
```

## Optional: run two tasks from one file

Goal: verify that `inspect-mlflow` records separate runs when one eval file contains multiple tasks.

```bash
uv run inspect eval examples/two_tasks.py
```

This file defines two small tasks (`hello_world_a`, `hello_world_b`). Inspect runs both tasks, and `inspect-mlflow` logs one MLflow run per task.

## Optional: run with two models

Goal: verify that `inspect-mlflow` records separate runs when a single eval is executed with multiple models.

```bash
uv run inspect eval examples/hello_world.py --model openai/gpt-4o-mini,openai/gpt-4o
```

This runs the same eval with two models and logs one MLflow run per model.
