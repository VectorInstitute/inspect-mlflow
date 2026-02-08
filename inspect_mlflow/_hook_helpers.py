"""Helper functions used by MLflow hooks."""

from __future__ import annotations

from typing import Any

from inspect_ai.scorer import CORRECT

from ._utils import (
    TAG_PREFIX,
    _clean_token,
    _coerce_metric,
    _iter_scores,
    _jsonable,
    _obj_get,
    _to_json,
)


def select_accuracy_score(
    sample: Any,
    *,
    preferred_scorer: str | None = None,
    task_scorers: list[str] | None = None,
) -> tuple[str | None, Any | None]:
    """Select scorer value used for aggregate accuracy.

    Selection order:
    1) Explicitly configured scorer name (preferred_scorer)
    2) First scorer from task definition order (task_scorers)
    3) Single available sample score
    """
    scores = getattr(sample, "scores", None)
    if not scores:
        return None, None

    items = [(str(name), score) for name, score in _iter_scores(scores)]
    if not items:
        return None, None

    score_map = {name: score for name, score in items}

    if preferred_scorer:
        score = score_map.get(preferred_scorer)
        return (preferred_scorer, score) if score is not None else (None, None)

    if task_scorers:
        for scorer_name in task_scorers:
            score = score_map.get(str(scorer_name))
            if score is not None:
                return str(scorer_name), score

    if len(items) == 1:
        return items[0]

    return None, None


def is_selected_score_correct(
    sample: Any,
    *,
    preferred_scorer: str | None = None,
    task_scorers: list[str] | None = None,
) -> tuple[bool | None, str | None]:
    """Return correctness for selected scorer and selected scorer name.

    Returns:
      - (True/False, scorer_name) when a scorer could be selected
      - (None, None) when no scorer could be selected
    """
    scorer_name, score = select_accuracy_score(
        sample,
        preferred_scorer=preferred_scorer,
        task_scorers=task_scorers,
    )
    if score is None:
        return None, None

    value = getattr(score, "value", score)
    if value == CORRECT or value is True:
        return True, scorer_name
    metric_value = _coerce_metric(value)
    if metric_value is not None:
        return metric_value == 1.0, scorer_name
    return False, scorer_name


def task_scorer_names_from_spec(spec: Any) -> list[str]:
    """Extract scorer names from task spec if available."""
    scorers = _obj_get(spec, "scorers")
    if not isinstance(scorers, list):
        return []

    names: list[str] = []
    for scorer in scorers:
        name = _obj_get(scorer, "name") or _obj_get(scorer, "scorer")
        if name:
            names.append(str(name))
    return names


def ensure_experiment(mlflow: Any, name: str, client: Any | None = None) -> None:
    """Create or restore experiment by name using client APIs only."""
    if client is None:
        client = mlflow.tracking.MlflowClient()
    existing = client.get_experiment_by_name(name)

    if existing is not None and getattr(existing, "lifecycle_stage", None) == "deleted":
        client.restore_experiment(existing.experiment_id)
        existing = client.get_experiment(existing.experiment_id)

    if existing is None:
        client.create_experiment(name)


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


def get_task_name(_data: Any, log: Any) -> str | None:
    """Extract task name from TaskEnd log metadata."""
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
        try:
            value = getattr(output, key, None)
        except Exception:
            value = None
        if value:
            return str(value)

    # Some output objects expose .message as a property that raises when there
    # are no choices (e.g., cancelled model generations).
    message = None
    try:
        message = getattr(output, "message", None)
    except Exception:
        message = None

    if message is not None:
        for key in ("content", "text"):
            value = _obj_get(message, key)
            if value:
                return str(value)

    # Fallback for ModelOutput-like objects that carry choices directly.
    choices = None
    try:
        choices = getattr(output, "choices", None)
    except Exception:
        choices = None

    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        choice_message = _obj_get(first_choice, "message")
        if choice_message is not None:
            for key in ("content", "text"):
                value = _obj_get(choice_message, key)
                if value:
                    return str(value)
        for key in ("text", "completion"):
            value = _obj_get(first_choice, key)
            if value:
                return str(value)

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
