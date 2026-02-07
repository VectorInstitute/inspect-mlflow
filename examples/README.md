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

For model selection, pass `--model` on commands below, or set `INSPECT_EVAL_MODEL` (in `.env` or exported env) and omit `--model`.
See option/env mapping docs: https://inspect.aisi.org.uk/options.html#specifying-options

Install provider dependencies for your chosen model before running examples (for example: `openai`, `anthropic`, `google-genai`, `aioboto3` for Bedrock).
See provider docs: https://inspect.aisi.org.uk/providers.html

In this tutorial, commands use OpenAI model names (`openai/gpt-4o-mini`, `openai/gpt-4o`) as examples. You can replace them with any supported model names for your setup.

Then run any of the included evals:

```bash
uv run inspect eval examples/hello_world.py --model openai/gpt-4o-mini
uv run inspect eval examples/parallel_add.py --model openai/gpt-4o-mini
uv run inspect eval examples/surfer.py --model openai/gpt-4o-mini
```

### Optional: run two tasks from one file

Goal: verify that `inspect-mlflow` records separate runs when one eval file contains multiple tasks.

```bash
uv run inspect eval examples/two_tasks.py --model openai/gpt-4o-mini
```

This file defines two small tasks (`hello_world_a`, `hello_world_b`). Inspect runs both tasks, and `inspect-mlflow` logs one MLflow run per task.

### Optional: run with two models

Goal: verify that `inspect-mlflow` records separate runs when a single eval is executed with multiple models.

```bash
uv run inspect eval examples/hello_world.py --model openai/gpt-4o-mini,openai/gpt-4o
```

This runs the same eval with two models and logs one MLflow run per model.

### Optional: run with eval-set

Goal: verify that `inspect-mlflow` groups related runs with a shared eval-set ID.

```bash
uv run inspect eval-set examples/two_tasks.py --id smoke-eval-set --model openai/gpt-4o-mini --log-dir logs/smoke-eval-set
```

This runs tasks as an eval-set and tags each MLflow run with `inspect.eval_set_id=smoke-eval-set`.
You can combine this with multiple models as well.
Use a fresh `--log-dir` for eval-set runs, or pass `--log-dir-allow-dirty` if you intentionally want to reuse a mixed log directory.

## Testing checklist

Use the examples above as integration smoke tests for `inspect-mlflow`.

After running an example, check MLflow UI (`http://127.0.0.1:5000`):

Note: to inspect run artifacts, switch the experiment type dropdown to `Machine learning` (not `GenAI apps & agents`), then open a run details page. The GenAI view focuses on traces and may not show the `Artifacts` panel.

1. Confirm new runs appear under the `inspect-mlflow` experiment.
2. Confirm key tags exist on runs: `inspect.task`, `inspect.model`, `inspect.status`.
3. Confirm summary metrics exist on each run (for example `inspect.samples_total`, `inspect.accuracy`).
4. Confirm artifacts exist under `inspect/` (for example `samples.json`, `tasks.json`, `events.json` when artifact logging is enabled).
5. Confirm run split behavior for `examples/two_tasks.py`: one run per task.
6. Confirm run split behavior for `--model model_a,model_b`: one run per model.
7. For `inspect eval-set`, confirm each run includes `inspect.eval_set_id`.
8. For `inspect eval-set --id smoke-eval-set`, filter MLflow runs by `tags.inspect.eval_set_id = "smoke-eval-set"` and confirm the grouped runs are from that command.
