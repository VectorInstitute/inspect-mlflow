"""State initialization and reset helpers for MLflow hooks."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, cast

from .config import MLflowSettings


def initialize_tracking_state(hook: Any) -> None:
    """Initialize eval-set, run, and per-task state containers on a hook."""
    # Eval-set level info (persists across retries within an eval-set)
    hook._eval_set_id = None
    hook._eval_set_log_dir = None
    hook._eval_set_run_count = 0

    # Run-level info (from RunStart, shared across tasks in a single run)
    hook._experiment_name = None
    hook._all_task_names = None

    # Per-task tracking (keyed by eval_id for parallel execution support)
    hook._active_runs = {}  # eval_id -> mlflow_run_id
    hook._task_names_by_eval_id = {}
    hook._task_sample_counts = {}  # eval_id -> sample count
    hook._task_scored_counts = {}  # eval_id -> accuracy-denominator sample count
    hook._task_correct_counts = {}  # eval_id -> correct count
    hook._task_sample_steps = {}  # eval_id -> step counter
    hook._task_disabled_eval_ids = set()  # eval_id values disabled via task metadata
    hook._task_scorer_names_by_eval_id = {}  # eval_id -> scorer names from task spec
    hook._task_models = defaultdict(set)  # eval_id -> models
    hook._task_raw_scores = defaultdict(lambda: defaultdict(Counter))
    hook._task_usage_totals = defaultdict(lambda: defaultdict(dict))
    hook._task_experiment_ids = {}
    hook._task_settings = cast(dict[str, MLflowSettings], {})
    hook._mlflow_client = None

    # Table rows for batch logging (per eval_id)
    hook._task_sample_rows = defaultdict(list)
    hook._task_message_rows = defaultdict(list)
    hook._task_sample_score_rows = defaultdict(list)
    hook._task_rows_data = defaultdict(list)
    hook._task_event_rows = defaultdict(list)
    hook._task_usage_rows = defaultdict(list)


def reset_run_state(hook: Any) -> None:
    """Clear run-level and per-task state after a run ends.

    Eval-set state is intentionally not reset here.
    """
    hook._active_runs.clear()
    hook._inspect_run_id = None
    hook._experiment_name = None
    hook._all_task_names = None
    hook._run_logging_enabled = False
    hook._mlflow_client = None

    # Clear per-task state
    hook._task_sample_counts.clear()
    hook._task_scored_counts.clear()
    hook._task_correct_counts.clear()
    hook._task_sample_steps.clear()
    hook._task_disabled_eval_ids.clear()
    hook._task_names_by_eval_id.clear()
    hook._task_scorer_names_by_eval_id.clear()
    hook._task_models.clear()
    hook._task_raw_scores.clear()
    hook._task_usage_totals.clear()
    hook._task_experiment_ids.clear()
    hook._task_settings.clear()
    hook._task_sample_rows.clear()
    hook._task_message_rows.clear()
    hook._task_sample_score_rows.clear()
    hook._task_rows_data.clear()
    hook._task_event_rows.clear()
    hook._task_usage_rows.clear()


def clear_task_state(
    hook: Any, eval_id: str, *, clear_run_tracking: bool = True
) -> None:
    """Clear per-task state for a single eval_id.

    When ``clear_run_tracking`` is False, keep ``_active_runs[eval_id]`` so
    orphaned runs can still be terminated later in ``on_run_end``.
    """
    if clear_run_tracking:
        hook._active_runs.pop(eval_id, None)

    hook._task_settings.pop(eval_id, None)
    hook._task_names_by_eval_id.pop(eval_id, None)
    hook._task_sample_counts.pop(eval_id, None)
    hook._task_scored_counts.pop(eval_id, None)
    hook._task_correct_counts.pop(eval_id, None)
    hook._task_sample_steps.pop(eval_id, None)
    hook._task_disabled_eval_ids.discard(eval_id)
    hook._task_scorer_names_by_eval_id.pop(eval_id, None)
    hook._task_models.pop(eval_id, None)
    hook._task_raw_scores.pop(eval_id, None)
    hook._task_usage_totals.pop(eval_id, None)
    hook._task_experiment_ids.pop(eval_id, None)
    hook._task_sample_rows.pop(eval_id, None)
    hook._task_message_rows.pop(eval_id, None)
    hook._task_sample_score_rows.pop(eval_id, None)
    hook._task_rows_data.pop(eval_id, None)
    hook._task_event_rows.pop(eval_id, None)
    hook._task_usage_rows.pop(eval_id, None)
