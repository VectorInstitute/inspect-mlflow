"""Helper functions used by MLflow hooks."""

from __future__ import annotations

from typing import Any

from inspect_ai.scorer import CORRECT

from ._utils import (
    TAG_PREFIX,
    _clean_token,
    _iter_scores,
    _jsonable,
    _obj_get,
    _to_json,
)


def is_correct(sample: Any) -> bool:
    """Check if a sample is correct based on scores."""
    scores = getattr(sample, "scores", None)
    if not scores:
        return False
    for _, score in _iter_scores(scores):
        value = getattr(score, "value", score)
        if value in {CORRECT, True}:
            return True
    return False


def ensure_experiment(mlflow: Any, name: str) -> None:
    """Create or get experiment by name and activate it."""
    client = mlflow.tracking.MlflowClient()
    existing = client.get_experiment_by_name(name)

    if existing is not None and getattr(existing, "lifecycle_stage", None) == "deleted":
        client.restore_experiment(existing.experiment_id)
        existing = client.get_experiment(existing.experiment_id)

    if existing is None:
        experiment_id = client.create_experiment(name)
    else:
        experiment_id = existing.experiment_id

    mlflow.set_experiment(experiment_id=experiment_id)


def default_experiment_name(
    run_id: str | None,
    task_names: list[str] | None,
) -> str:
    """Generate default experiment name."""
    base = "eval"
    if task_names:
        base = _clean_token(str(task_names[0]), max_len=48)
    if run_id:
        return f"{TAG_PREFIX}-{base}-{run_id[:8]}"
    return f"{TAG_PREFIX}-{base}"


def get_task_name(data: Any, log: Any) -> str | None:
    """Extract task name from TaskEnd data or log."""
    spec = getattr(data, "spec", None)
    if spec:
        name = getattr(spec, "task", None) or getattr(spec, "name", None)
        if name:
            return str(name)

    eval_info = _obj_get(log, "eval")
    if eval_info:
        for key in ("task_display_name", "task", "task_registry_name"):
            value = _obj_get(eval_info, key)
            if value:
                return str(value)
    return None


def get_sample_output_text(sample: Any) -> str | None:
    """Extract output text from sample."""
    if sample is None:
        return None
    output = getattr(sample, "output", None)
    if output is None:
        return None
    for key in ("completion", "text"):
        value = getattr(output, key, None)
        if value:
            return str(value)
    message = getattr(output, "message", None)
    if message:
        content = getattr(message, "content", None)
        if content:
            return str(content)
    return _to_json(output)


def scores_to_dict(scores: Any) -> dict[str, Any]:
    """Convert scores to JSON-serializable dict."""
    output: dict[str, Any] = {}
    for name, score in _iter_scores(scores):
        output[str(name)] = _jsonable(score)
    return output


def rows_to_columns(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Convert list of row dicts to column dict (for MLflow tables)."""
    columns: dict[str, list[Any]] = {}
    for row in rows:
        for key in row.keys():
            columns.setdefault(str(key), [])
    for row in rows:
        for key in columns.keys():
            columns[key].append(row.get(key))
    return columns
