"""Logging mixin for MLflow hooks."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from ._utils import (
    TAG_PREFIX,
    _clean_key,
    _clean_token,
    _coerce_metric,
    _iter_scores,
    _location_to_local_path,
    _obj_get,
    _sum_usage,
    _to_json,
    _usage_to_dict,
)

_LOG = logging.getLogger(__name__)


class LoggingMixin:
    """Mixin providing MLflow logging methods for MLflowHooks.

    Expects the host class to provide:
    - self._task_raw_scores: dict[str, dict[tuple[str, str], Counter[str]]]
    - self._task_sample_score_rows: dict[str, list[dict[str, Any]]]
    - self._task_sample_rows: dict[str, list[dict[str, Any]]]
    - self._task_rows_data: dict[str, list[dict[str, Any]]]
    - self._task_event_rows: dict[str, list[dict[str, Any]]]
    - self._task_usage_rows: dict[str, list[dict[str, Any]]]
    - self._task_models: dict[str, set[str]]
    - self._task_usage_totals: dict[str, dict[str, dict[str, int]]]
    - self._get_sample_output_text(sample) -> str | None
    - self._scores_to_dict(scores) -> dict
    - self._rows_to_columns(rows) -> dict (static)
    """

    # -------------------------------------------------------------------------
    # Metrics and Parameters
    # -------------------------------------------------------------------------

    # Fields to skip when logging EvalSpec — either irrelevant or logged separately.
    _SPEC_SKIP_FIELDS: set[str] = {
        "created",
        "run_id",
        "eval_id",  # identifiers logged elsewhere / separately
        "packages",
        "revision",
        "metadata",  # noisy / large
        "model_config",  # pydantic internal
    }

    def _log_task_params_client(
        self, client: Any, run_id: str, spec: Any, eval_id: str | None
    ) -> None:
        """Log task specification as MLflow parameters using client API.

        Dynamically iterates over every field on the EvalSpec (and nested
        EvalConfig / EvalDataset) so new fields added by Inspect are picked
        up automatically.
        """
        params: dict[str, str] = {}

        # Always log eval_id explicitly (not on spec during TaskStart)
        if eval_id:
            params["eval_id"] = str(eval_id)

        # Dynamically dump all spec fields
        try:
            spec_data = spec.model_dump(exclude_none=True)
        except Exception:
            spec_data = {}

        for key, value in spec_data.items():
            if key in self._SPEC_SKIP_FIELDS:
                continue

            if key == "config" and isinstance(value, dict):
                # Flatten eval config fields (limit, epochs, max_samples, …)
                for ck, cv in value.items():
                    params[ck] = _to_json(cv) or str(cv)
            elif key == "dataset" and isinstance(value, dict):
                # Prefix dataset sub-fields to avoid collisions
                for dk, dv in value.items():
                    if dk == "name":
                        params["dataset"] = str(dv)
                    else:
                        params[f"dataset_{dk}"] = _to_json(dv) or str(dv)
            elif isinstance(value, (dict, list)):
                serialized = _to_json(value)
                if serialized:
                    params[key] = serialized
            else:
                params[key] = str(value)

        for key, value in params.items():
            try:
                client.log_param(run_id, _clean_key(key), value)
            except Exception:
                pass

    def _log_sample_scores_client(
        self,
        client: Any,
        run_id: str,
        eval_id: str,
        task_name: str,
        scores: Any,
        step: int,
    ) -> None:
        """Log individual sample scores using client API."""
        for name, score in _iter_scores(scores):
            raw_value = getattr(score, "value", score)
            metric_value = _coerce_metric(raw_value)

            # Track raw score distribution for this task
            self._task_raw_scores[eval_id][(task_name, str(name))][str(raw_value)] += 1

            # Record for table
            self._task_sample_score_rows[eval_id].append(
                {
                    "task_name": task_name,
                    "eval_id": eval_id,
                    "scorer": str(name),
                    "raw_value": str(raw_value),
                    "numeric_value": metric_value,
                }
            )

            # Log as metric if numeric
            if metric_value is not None:
                metric_name = _clean_key(f"{TAG_PREFIX}.sample.{eval_id}.{name}")
                client.log_metric(run_id, metric_name, metric_value, step=step)

    def _log_task_scores_client(
        self, client: Any, run_id: str, log: Any, task_name: str
    ) -> None:
        """Log task-level aggregate scores using client API."""
        scores = _obj_get(log, "scores") or _obj_get(log, "metrics")
        if scores:
            for name, score in _iter_scores(scores):
                metric_value = _coerce_metric(score)
                if metric_value is not None:
                    metric_name = _clean_key(f"{TAG_PREFIX}.task.{task_name}.{name}")
                    client.log_metric(run_id, metric_name, metric_value)
            return

        results = _obj_get(log, "results")
        if results:
            results_scores = _obj_get(results, "scores")
            if results_scores:
                for score in results_scores:
                    score_name = _obj_get(score, "name") or "score"
                    metrics = _obj_get(score, "metrics") or {}
                    for metric_name, metric in metrics.items():
                        value = _obj_get(metric, "value")
                        metric_value = _coerce_metric(value)
                        if metric_value is not None:
                            key = _clean_key(
                                f"{TAG_PREFIX}.task.{task_name}.{score_name}.{metric_name}"
                            )
                            client.log_metric(run_id, key, metric_value)

    def _aggregate_usage_for_task(
        self, eval_id: str, stats_usage: dict[str, Any]
    ) -> None:
        """Aggregate usage from task stats for a specific eval_id."""
        for model_key, usage in stats_usage.items():
            usage_dict = _usage_to_dict(usage)
            if not usage_dict:
                continue
            self._task_models[eval_id].add(str(model_key))
            totals = self._task_usage_totals[eval_id].setdefault(str(model_key), {})
            for key, value in usage_dict.items():
                totals[key] = int(totals.get(key, 0)) + int(value)

    def _set_usage_totals_for_task(
        self, eval_id: str, stats_usage: dict[str, Any]
    ) -> None:
        """Set usage totals from task stats for a specific eval_id.

        This replaces current totals for the eval rather than adding to them.
        It is used at task end to avoid double-counting when per-sample usage
        has already been aggregated during sample events.
        """
        totals_for_eval: dict[str, dict[str, int]] = {}
        for model_key, usage in stats_usage.items():
            usage_dict = _usage_to_dict(usage)
            if not usage_dict:
                continue
            model_name = str(model_key)
            self._task_models[eval_id].add(model_name)
            totals_for_eval[model_name] = {
                key: int(value) for key, value in usage_dict.items()
            }

        if totals_for_eval:
            self._task_usage_totals[eval_id] = totals_for_eval

    def _log_usage_metrics_client(self, client: Any, run_id: str, eval_id: str) -> None:
        """Log aggregated token usage metrics using client API."""
        usage_totals = self._task_usage_totals.get(eval_id, {})
        if not usage_totals:
            return

        grand_totals: dict[str, int] = defaultdict(int)
        for model_name, usage in usage_totals.items():
            clean_model = _clean_token(model_name, max_len=64)
            for key, value in usage.items():
                grand_totals[key] += int(value)
                metric_key = _clean_key(f"{TAG_PREFIX}.usage.{clean_model}.{key}")
                client.log_metric(run_id, metric_key, float(value))

        for key, value in grand_totals.items():
            client.log_metric(
                run_id, _clean_key(f"{TAG_PREFIX}.tokens.{key}"), float(value)
            )

    # -------------------------------------------------------------------------
    # Table Recording
    # -------------------------------------------------------------------------

    def _record_task_row(self, eval_id: str, spec: Any) -> None:
        """Record task info for the tasks table (per eval_id)."""
        self._task_rows_data[eval_id].append(
            {
                "task_name": _obj_get(spec, "task") or _obj_get(spec, "name"),
                "eval_id": eval_id,
                "task_file": _obj_get(spec, "task_file"),
                "task_version": _obj_get(spec, "task_version"),
                "task_id": _obj_get(spec, "task_id"),
                "solver": str(_obj_get(spec, "solver"))
                if _obj_get(spec, "solver")
                else None,
                "model": str(_obj_get(spec, "model"))
                if _obj_get(spec, "model")
                else None,
                "dataset": str(_obj_get(_obj_get(spec, "dataset"), "name")),
                "dataset_size": _obj_get(_obj_get(spec, "dataset"), "size"),
            }
        )

    def _record_sample_row(
        self, eval_id: str, task_name: str, sample: Any, scores: Any
    ) -> None:
        """Record sample info for the samples table (per eval_id)."""
        sample_id = _obj_get(sample, "id")
        usage_map = _obj_get(sample, "model_usage")
        usage = _sum_usage(usage_map)
        events = _obj_get(sample, "events")

        self._task_sample_rows[eval_id].append(
            {
                "task_name": task_name,
                "eval_id": eval_id,
                "sample_id": sample_id,
                "input": _to_json(_obj_get(sample, "input")),
                "target": _to_json(_obj_get(sample, "target")),
                "output": self._get_sample_output_text(sample),
                "scores": self._scores_to_dict(scores),
                "events_count": len(events) if isinstance(events, list) else None,
                "total_time": _obj_get(sample, "total_time"),
                "working_time": _obj_get(sample, "working_time"),
                "error": _to_json(_obj_get(sample, "error")),
                **{f"usage_{k}": v for k, v in usage.items()},
            }
        )

    def _record_sample_usage(
        self, eval_id: str, task_name: str, sample_id: Any, sample: Any
    ) -> None:
        """Record per-sample usage for the usage table (per eval_id)."""
        usage_map = _obj_get(sample, "model_usage")
        if not isinstance(usage_map, dict):
            return

        for model_name, usage in usage_map.items():
            model_key = str(model_name)
            usage_dict = _usage_to_dict(usage)
            if not usage_dict:
                continue

            self._task_models[eval_id].add(model_key)
            self._task_usage_rows[eval_id].append(
                {
                    "task_name": task_name,
                    "eval_id": eval_id,
                    "sample_id": sample_id,
                    "model": model_key,
                    **usage_dict,
                }
            )

            # Aggregate totals for this task
            totals = self._task_usage_totals[eval_id].setdefault(model_key, {})
            for key, value in usage_dict.items():
                totals[key] = int(totals.get(key, 0)) + int(value)

    def _record_sample_events(
        self, eval_id: str, task_name: str, sample_id: Any, sample: Any
    ) -> None:
        """Record events for the events table."""
        events = _obj_get(sample, "events")
        if not isinstance(events, list):
            return

        for idx, event in enumerate(events):
            event_type = _obj_get(event, "event") or _obj_get(event, "type")
            row: dict[str, Any] = {
                "task_name": task_name,
                "eval_id": eval_id,
                "sample_id": sample_id,
                "event_index": idx,
                "event_type": str(event_type) if event_type else None,
                "timestamp": _to_json(_obj_get(event, "timestamp")),
            }

            if event_type == "model":
                row["model"] = _obj_get(event, "model")
                output = _obj_get(event, "output")
                row["completion"] = _to_json(_obj_get(output, "completion"))
                usage = _usage_to_dict(_obj_get(output, "usage"))
                for k, v in usage.items():
                    row[f"usage_{k}"] = v

            elif event_type == "tool":
                row["tool_function"] = _obj_get(event, "function")
                row["tool_arguments"] = _to_json(_obj_get(event, "arguments"))
                row["tool_result"] = _to_json(_obj_get(event, "result"))
                error = _obj_get(event, "error")
                if error:
                    row["tool_error"] = _to_json(error)

            elif event_type == "error":
                row["error"] = _to_json(_obj_get(event, "error"))

            self._task_event_rows[eval_id].append(row)

    # -------------------------------------------------------------------------
    # Artifact Logging
    # -------------------------------------------------------------------------

    def _log_tables_for_task(self, client: Any, run_id: str, eval_id: str) -> None:
        """Log all accumulated tables for a specific task as artifacts."""
        tables = [
            ("samples", self._task_sample_rows.get(eval_id, [])),
            ("sample_scores", self._task_sample_score_rows.get(eval_id, [])),
            ("tasks", self._task_rows_data.get(eval_id, [])),
            ("events", self._task_event_rows.get(eval_id, [])),
            ("model_usage", self._task_usage_rows.get(eval_id, [])),
        ]

        for name, rows in tables:
            if not rows:
                continue
            try:
                client.log_table(
                    run_id=run_id,
                    data=self._rows_to_columns(rows),
                    artifact_file=f"{TAG_PREFIX}/{name}.json",
                )
            except Exception:
                _LOG.debug(f"Failed to log {name} table", exc_info=True)

    def _log_task_inspect_logs(
        self,
        client: Any,
        run_id: str,
        log: Any,
        eval_id: str | None = None,
        task_id: str | None = None,
    ) -> None:
        """Upload Inspect log file for a specific task and log its path."""
        logged_paths: list[str] = []
        uploaded_paths: list[str] = []
        search_dirs: list[Path] = []

        def add_search_dir(path: Path) -> None:
            """Add directory once, preserving order."""
            if path.exists() and path.is_dir() and path not in search_dirs:
                search_dirs.append(path)

        # Get location from the log object
        if log is not None:
            location = _obj_get(log, "location")
            if location:
                location_str = str(location)
                logged_paths.append(location_str)

                local_path = _location_to_local_path(location_str)
                if local_path and local_path.exists():
                    add_search_dir(local_path.parent)
                    uploaded_paths.append(location_str)
                    try:
                        client.log_artifact(
                            run_id=run_id,
                            local_path=str(local_path),
                            artifact_path=f"{TAG_PREFIX}/logs",
                        )
                    except Exception:
                        _LOG.debug(f"Could not upload log: {local_path}", exc_info=True)

        # Prefer inspect-provided log dir if available (e.g., custom INSPECT_LOG_DIR).
        eval_set_log_dir = getattr(self, "_eval_set_log_dir", None)
        if isinstance(eval_set_log_dir, str) and eval_set_log_dir:
            local_eval_set_log_dir = _location_to_local_path(eval_set_log_dir)
            if local_eval_set_log_dir is not None:
                add_search_dir(local_eval_set_log_dir)

        # Fallback to default Inspect log directory under current working dir.
        add_search_dir(Path.cwd() / "logs")

        # Collect IDs to search for in logs directory
        search_ids: list[str] = []
        if task_id:
            search_ids.append(task_id)

        if log is not None:
            eval_info = _obj_get(log, "eval")
            eval_run_id = _obj_get(eval_info, "run_id")
            if eval_run_id and str(eval_run_id) not in search_ids:
                search_ids.append(str(eval_run_id))
            log_task_id = _obj_get(eval_info, "task_id")
            if log_task_id and str(log_task_id) not in search_ids:
                search_ids.append(str(log_task_id))

        # Search known log directories for matching files.
        if search_dirs and search_ids:
            for logs_dir in search_dirs:
                found_match = False
                for search_id in search_ids:
                    for pattern in [f"*_{search_id}.json", f"*_{search_id}.eval"]:
                        for p in logs_dir.glob(pattern):
                            path_str = str(p)
                            if path_str not in logged_paths:
                                logged_paths.append(path_str)
                            if path_str not in uploaded_paths:
                                uploaded_paths.append(path_str)
                                try:
                                    client.log_artifact(
                                        run_id=run_id,
                                        local_path=str(p),
                                        artifact_path=f"{TAG_PREFIX}/logs",
                                    )
                                except Exception:
                                    _LOG.debug(
                                        f"Could not upload log: {p}", exc_info=True
                                    )
                            found_match = True
                    if found_match:
                        break
                if found_match:
                    break

        # Log paths as params
        if logged_paths:
            for idx, path in enumerate(logged_paths):
                param_key = (
                    f"{TAG_PREFIX}.log_file"
                    if len(logged_paths) == 1
                    else f"{TAG_PREFIX}.log_file_{idx + 1}"
                )
                self._log_param_safe(client, run_id, param_key, path)

    @staticmethod
    def _log_param_safe(client: Any, run_id: str, key: str, value: str) -> None:
        """Log param, ignoring errors (e.g., duplicate keys)."""
        try:
            client.log_param(run_id, key, value)
        except Exception:
            _LOG.debug(f"Could not log param {key}", exc_info=True)
