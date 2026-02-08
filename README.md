# Inspect MLflow Extension

`inspect-mlflow` logs [Inspect](https://inspect.aisi.org.uk/) evaluation runs into [MLflow](https://mlflow.org/).

The intent is:

- Inspect handles eval execution, scoring, retries, and task logic.
- MLflow becomes the post-run system of record for analysis and tracking.
- You should not need to inspect raw `.json` / `.eval` logs for normal workflows.

At runtime, the extension consumes Inspect hook events and writes run data to MLflow.

## Install

```bash
pip install inspect-mlflow
```

For local development:

```bash
uv venv
uv pip install -e '.[dev]'
```

## Quick Start

1. Set up an MLflow tracking server.

You can use an existing remote server (self-hosted or managed) or run one locally.
For local development, run this SQLite-based server:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```
If your MLflow server requires authentication, configure access in your environment before running Inspect.

2. Set the tracking URI.

For the local server example above, this will be:

```bash
export INSPECT_MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

For a remote server, set it to your server URL.

3. Ensure `inspect-mlflow` is installed in the same environment where you run Inspect (see [Install](#install)), then run an eval:

```bash
inspect eval path/to/task.py
```

For local testing / runnable quick-start commands, see [examples/README.md](examples/README.md).
Other environment variables are documented in [Configuration](#configuration).

When `inspect-mlflow` is installed in the Inspect runtime environment, MLflow logging is enabled by default (`INSPECT_MLFLOW_ENABLED=true`) and runs are logged automatically.
If you need to disable logging, set:

```bash
export INSPECT_MLFLOW_ENABLED=false
```

4. Open MLflow and review runs.

Note: in MLflow 3.x, to inspect run artifacts, switch experiment type to `Machine learning` (not `GenAI apps & agents`) and open a run details page.

## Where Data Is Stored

`inspect-mlflow` sends run data to the MLflow server at `INSPECT_MLFLOW_TRACKING_URI`.
The MLflow server decides where data is stored.

- Run metadata (params, metrics, tags, run state) is stored in the server backend store.
- Artifacts are stored in the server artifact store.

For the Quick Start local server command above (`mlflow ui --backend-store-uri sqlite:///mlflow.db ...`):

- Backend store is `sqlite:///mlflow.db` (saved in `mlflow.db`).
- Artifacts are served from the default local artifact location (typically `./mlartifacts`).

You can change storage locations by configuring MLflow server options such as `--backend-store-uri`, `--artifacts-destination`, and `--default-artifact-root`.

## Run Mapping

- One MLflow run is created per Inspect task execution (`eval_id`).
- Multi-task or multi-model runs create multiple MLflow runs.
- `inspect eval-set` adds grouping via tag `inspect.eval_set_id`.

Example eval-set run:

```bash
inspect eval-set examples/two_tasks.py \
  --id smoke-eval-set \
  --model openai/gpt-4o-mini \
  --log-dir logs/smoke-eval-set
```

## Configuration

Settings can be provided by:

- environment variables (`INSPECT_MLFLOW_*`)
- per-task metadata overrides (metadata keys prefixed with `inspect_mlflow_`). Environment variables are the primary configuration path; per-task metadata is for targeted overrides.

The sections below group configuration into:

- Core settings
- Logging controls
- Autolog behavior
- Task metadata overrides

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `INSPECT_MLFLOW_ENABLED` | `true` | Enable/disable MLflow logging |
| `INSPECT_MLFLOW_TRACKING_URI` | - | MLflow tracking server URI |
| `INSPECT_MLFLOW_EXPERIMENT` | auto | Experiment name (auto-generated if unset) |
| `INSPECT_MLFLOW_RUN_NAME` | auto | Run name (`<task>-<model>-<eval_id>` unless overridden) |
| `INSPECT_MLFLOW_ACCURACY_SCORER` | auto | Scorer name used for `inspect.accuracy`/`inspect.samples_correct` (defaults to first task scorer when available) |

### Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `INSPECT_MLFLOW_LOG_ARTIFACTS` | `true` | Log artifacts (tables + Inspect logs) |
| `INSPECT_MLFLOW_LOG_TRACES` | `true` | Enable sample/event trace logging |

### Autolog

| Variable | Default | Description |
|----------|---------|-------------|
| `INSPECT_MLFLOW_AUTOLOG_ENABLED` | `true` | Enable MLflow autolog for LLM libraries |
| `INSPECT_MLFLOW_AUTOLOG_MODELS` | `openai,anthropic,langchain,litellm` | CSV or JSON array of libraries to autolog |

Autolog entries are enabled only when both MLflow flavor support and provider dependencies are available.

### Task Metadata Overrides

Use task metadata overrides when one specific task needs behavior different from your global environment-based defaults.
Each override key should use the `inspect_mlflow_` prefix.

Current support is split by scope:

- Per-task overrides (applied in `on_task_start` / task lifecycle): `enabled`, `experiment`, `run_name`, `log_artifacts`, `log_traces`, `accuracy_scorer`
- Run-level only (read once in `on_run_start`): `tracking_uri`, `autolog_enabled`, `autolog_models`

```python
@task
def my_task():
    return Task(
        dataset=...,
        solver=...,
        metadata={
            "inspect_mlflow_enabled": False,
            "inspect_mlflow_experiment": "my-custom-experiment",
            "inspect_mlflow_accuracy_scorer": "exact",
        },
    )
```

If you need different tracking URIs or autolog settings, run those tasks in separate Inspect runs with different environment variables.

### Accuracy Semantics

`inspect-mlflow` computes `inspect.accuracy` and `inspect.samples_correct` from one selected scorer per task:

1. `INSPECT_MLFLOW_ACCURACY_SCORER` / `inspect_mlflow_accuracy_scorer` when set.
2. Otherwise, the first scorer declared on the Inspect task.
3. Otherwise, if a sample has exactly one score, that score is used.

If no scorer can be selected for a sample, it is excluded from accuracy denominator (`inspect.samples_scored`).
If a task has zero scored samples, `inspect.accuracy` and `inspect.samples_correct` are not emitted.

## What Gets Logged

### Tags

- `inspect.task`
- `inspect.model`
- `inspect.status` (`FINISHED` / `FAILED`)
- `inspect.eval_set_id` (when running eval-set)

### Parameters

- `eval_id`
- flattened task/eval spec fields (model, solver, dataset, config values)
- `inspect.log_file` / `inspect.log_file_<N>` when log file paths are available

### Metrics

- running: `inspect.samples`, `inspect.samples_scored`, `inspect.samples_correct`, `inspect.accuracy`
- final: `inspect.samples_total`, `inspect.samples_scored`, `inspect.samples_correct`, `inspect.accuracy`
- sample scores: `inspect.sample.<eval_id>.<scorer>`
- task aggregate scores: `inspect.task.<task>.<scorer>[.<metric>]`
- token usage: `inspect.usage.<model>.<token_field>`, `inspect.tokens.<token_field>`

### Artifacts

When `INSPECT_MLFLOW_LOG_ARTIFACTS=true`:

- `inspect/samples.json`
- `inspect/messages.json`
- `inspect/sample_scores.json`
- `inspect/tasks.json`
- `inspect/events.json`
- `inspect/model_usage.json`
- `inspect/logs/<log-file>` (best effort)

### Traces

When `INSPECT_MLFLOW_LOG_TRACES=true`:

- each sample creates a root trace span with sample input/output
- each chat message in `sample.messages` is emitted as a child span (`message.<idx>.<role>`)
- model/tool/error events are emitted as child spans

## Examples

Example workflows are in `examples/README.md`.

Examples are integration smoke tests for this extension and are based on Inspect examples, not a separate benchmark suite.
