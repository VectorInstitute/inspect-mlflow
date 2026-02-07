"""MLflow hooks for Inspect evaluation logging."""

from __future__ import annotations

import importlib.util
import logging
import threading
from typing import Any

from inspect_ai.hooks import (
    EvalSetEnd,
    EvalSetStart,
    Hooks,
    RunEnd,
    RunStart,
    SampleEnd,
    TaskEnd,
    TaskStart,
    hooks,
)

from ._autolog import enable_autolog
from ._hook_helpers import (
    default_experiment_name,
    ensure_experiment,
    get_sample_output_text,
    get_task_name,
    is_correct,
    rows_to_columns,
    scores_to_dict,
)
from ._logging import LoggingMixin
from ._state import initialize_tracking_state, reset_run_state
from ._tracing import TracingMixin
from ._utils import (
    TAG_PREFIX,
    _obj_get,
)
from .config import MLflowSettings

_LOG = logging.getLogger(__name__)


@hooks(name="mlflow", description="Log Inspect evaluations to MLflow with rich tracing")
class MLflowHooks(Hooks, TracingMixin, LoggingMixin):
    """MLflow integration hooks for Inspect."""

    def __init__(self) -> None:
        self._settings: MLflowSettings | None = None
        self._lock = threading.Lock()
        self._inspect_run_id: str | None = None
        self._run_logging_enabled: bool = False
        self._autolog_enabled: bool = False
        self._trace_supported: bool = True
        initialize_tracking_state(self)

    @property
    def settings(self) -> MLflowSettings:
        """Get current settings, loading from env if needed."""
        if self._settings is None:
            self._settings = MLflowSettings()
        return self._settings

    def enabled(self) -> bool:
        """Check if MLflow logging is enabled."""
        return self.settings.enabled

    # -------------------------------------------------------------------------
    # Lifecycle Hooks
    # -------------------------------------------------------------------------

    async def on_eval_set_start(self, data: EvalSetStart) -> None:
        """Track eval-set context for grouping runs across retries.

        An eval-set wraps one or more eval() invocations (with retry logic).
        We store the eval_set_id so it can be tagged on every MLflow run,
        letting users filter/group all runs from the same eval-set.
        """
        if not self.enabled():
            return

        self._eval_set_id = str(getattr(data, "eval_set_id", "") or "")
        self._eval_set_log_dir = str(getattr(data, "log_dir", "") or "")
        self._eval_set_run_count = 0

        _LOG.info(f"Eval set started: {self._eval_set_id}")

    async def on_eval_set_end(self, data: EvalSetEnd) -> None:
        """Clean up eval-set context."""
        if not self.enabled():
            return

        _LOG.info(
            f"Eval set ended: {self._eval_set_id} ({self._eval_set_run_count} run(s))"
        )

        self._eval_set_id = None
        self._eval_set_log_dir = None
        self._eval_set_run_count = 0

    async def on_run_start(self, data: RunStart) -> None:
        """Initialize MLflow tracking and enable autolog integrations.

        Note: We don't start a run here. Each task gets its own MLflow run,
        started in on_task_start. This allows multi-model evaluations to
        create separate runs for easy comparison.
        """
        if not self.enabled():
            return

        try:
            mlflow = self._mlflow()
            cfg = self.settings

            # Configure tracking URI
            if cfg.tracking_uri:
                mlflow.set_tracking_uri(cfg.tracking_uri)

            # Enable async logging (graceful fallback for older MLflow versions)
            try:
                mlflow.config.enable_async_logging(True)
            except Exception:
                _LOG.debug("Async logging not available in this MLflow version")

            # Enable autolog for LLM libraries
            if cfg.autolog_enabled:
                self._enable_autolog(mlflow, cfg.autolog_models)

            # Store run-level info for use by tasks
            self._inspect_run_id = str(getattr(data, "run_id", "") or "")
            self._all_task_names = getattr(data, "task_names", None)
            self._eval_set_run_count += 1

            # Pick up eval_set_id from RunStart if not already set by on_eval_set_start
            run_eval_set_id = str(getattr(data, "eval_set_id", "") or "")
            if run_eval_set_id and not self._eval_set_id:
                self._eval_set_id = run_eval_set_id

            # Determine experiment name (shared across all tasks in this run)
            self._experiment_name = cfg.experiment or self._default_experiment_name(
                run_id=self._inspect_run_id,
                task_names=self._all_task_names,
            )
            self._ensure_experiment(mlflow, self._experiment_name)

            # Mark that we're ready to log (runs will be started per-task)
            self._run_logging_enabled = True

            _LOG.info(
                f"MLflow tracking initialized for experiment: {self._experiment_name}"
            )

        except Exception:
            _LOG.exception("MLflow hook failed during run start")
            self._run_logging_enabled = False

    async def on_task_start(self, data: TaskStart) -> None:
        """Start a new MLflow run for this task and log configuration.

        Each task (which may use a different model) gets its own MLflow run.
        This enables easy comparison of models in the MLflow UI.

        Note: Tasks may run in parallel, so we track runs by eval_id.
        """
        if not self.enabled() or not self._run_logging_enabled:
            return

        try:
            eval_id = str(getattr(data, "eval_id", None) or "unknown")
            task_settings = self.settings

            # Check for task metadata overrides
            spec = getattr(data, "spec", None)
            if spec is not None:
                metadata = getattr(spec, "metadata", None)
                if metadata:
                    task_settings = MLflowSettings.from_metadata(metadata)
                    if not task_settings.enabled:
                        _LOG.info("MLflow disabled via task metadata")
                        return

            mlflow = self._mlflow()
            cfg = task_settings

            task_name = (
                getattr(spec, "task", None) or getattr(spec, "name", None)
                if spec
                else None
            )
            model_name = str(getattr(spec, "model", None) or "") if spec else ""

            # Store task name for lookup
            if task_name:
                self._task_names_by_eval_id[eval_id] = str(task_name)

            # Generate run name: task_name-model-eval_id (matches Inspect log naming)
            if cfg.run_name:
                run_name = cfg.run_name
            else:
                parts = []
                if task_name:
                    parts.append(str(task_name))
                if model_name:
                    short_model = (
                        model_name.split("/")[-1] if "/" in model_name else model_name
                    )
                    parts.append(short_model)
                parts.append(eval_id)
                run_name = "-".join(parts) if parts else self._inspect_run_id

            # Initialize per-task state
            self._task_sample_counts[eval_id] = 0
            self._task_correct_counts[eval_id] = 0
            self._task_sample_steps[eval_id] = 0

            # Get experiment ID
            experiment_name = cfg.experiment or self._experiment_name
            if experiment_name is None:
                experiment_name = self._default_experiment_name(
                    run_id=self._inspect_run_id, task_names=self._all_task_names
                )
            self._ensure_experiment(mlflow, experiment_name)

            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name(experiment_name)
            experiment_id = experiment.experiment_id if experiment else "0"

            # Create a new run using the client (avoids affecting global active run)
            with self._lock:
                run = client.create_run(
                    experiment_id=experiment_id,
                    run_name=run_name,
                )
                run_id = run.info.run_id
                self._active_runs[eval_id] = run_id
                self._task_settings[eval_id] = cfg

            # Tags: only things useful for filtering/grouping in the UI
            if task_name:
                client.set_tag(run_id, f"{TAG_PREFIX}.task", str(task_name))
            if model_name:
                client.set_tag(run_id, f"{TAG_PREFIX}.model", model_name)
            if self._eval_set_id:
                client.set_tag(run_id, f"{TAG_PREFIX}.eval_set_id", self._eval_set_id)

            # Log task parameters
            if spec is not None:
                self._log_task_params_client(client, run_id, spec, eval_id)

            # Track task in table
            if spec is not None:
                self._record_task_row(eval_id, spec)

            _LOG.info(
                f"MLflow run started for task {task_name} (eval_id={eval_id}): {run_id}"
            )

        except Exception:
            _LOG.exception("MLflow hook failed during task start")

    async def on_sample_end(self, data: SampleEnd) -> None:
        """Log sample results with hierarchical tracing."""
        if not self.enabled() or not self._run_logging_enabled:
            return

        eval_id = str(getattr(data, "eval_id", "eval"))
        run_id = self._active_runs.get(eval_id)
        if not run_id:
            return

        try:
            mlflow = self._mlflow()
            cfg = self._task_settings.get(eval_id, self.settings)
            client = mlflow.tracking.MlflowClient()

            sample = getattr(data, "sample", None)
            scores = getattr(sample, "scores", None) if sample is not None else None

            # Get per-task step
            step = self._task_sample_steps.get(eval_id, 0)
            self._task_sample_steps[eval_id] = step + 1

            # Update per-task accuracy counters
            with self._lock:
                self._task_sample_counts[eval_id] = (
                    self._task_sample_counts.get(eval_id, 0) + 1
                )
                if sample is not None and self._is_correct(sample):
                    self._task_correct_counts[eval_id] = (
                        self._task_correct_counts.get(eval_id, 0) + 1
                    )

            total_samples = self._task_sample_counts.get(eval_id, 0)
            correct_samples = self._task_correct_counts.get(eval_id, 0)
            accuracy = correct_samples / total_samples if total_samples > 0 else 0.0

            # Log progress metrics using client API (works with parallel tasks)
            client.log_metric(run_id, f"{TAG_PREFIX}.samples", total_samples, step=step)
            client.log_metric(run_id, f"{TAG_PREFIX}.accuracy", accuracy, step=step)
            client.log_metric(
                run_id, f"{TAG_PREFIX}.samples_correct", correct_samples, step=step
            )

            # Log sample scores
            task_name = self._task_names_by_eval_id.get(eval_id, eval_id)
            if scores:
                self._log_sample_scores_client(
                    client, run_id, eval_id, task_name, scores, step
                )

            # Record sample data for tables
            self._record_sample_row(eval_id, task_name, sample, scores)

            # Record usage and events
            sample_id = _obj_get(sample, "id")
            self._record_sample_usage(eval_id, task_name, sample_id, sample)
            self._record_sample_events(eval_id, task_name, sample_id, sample)

            # Log hierarchical trace - activate the specific run context
            if cfg.log_traces:
                try:
                    trace_id = self._log_sample_trace(
                        mlflow, task_name, eval_id, sample
                    )
                    if trace_id:
                        try:
                            client.link_traces_to_run(
                                trace_ids=[trace_id], run_id=run_id
                            )
                        except Exception:
                            _LOG.debug("Could not link trace to run", exc_info=True)
                except Exception:
                    _LOG.debug("Could not log trace", exc_info=True)

        except Exception:
            _LOG.exception("MLflow hook failed during sample end")

    async def on_task_end(self, data: TaskEnd) -> None:
        """Finalize and end the MLflow run for this task.

        Each task has its own MLflow run, so we finalize metrics,
        log artifacts, and end the run here.
        """
        if not self.enabled() or not self._run_logging_enabled:
            return

        eval_id = str(getattr(data, "eval_id", "eval"))
        run_id = self._active_runs.get(eval_id)
        if not run_id:
            _LOG.warning(f"No active run found for eval_id={eval_id}")
            return

        try:
            mlflow = self._mlflow()
            cfg = self._task_settings.get(eval_id, self.settings)
            client = mlflow.tracking.MlflowClient()

            log = getattr(data, "log", None)
            task_name = self._get_task_name(data, log) or eval_id

            # Get task_id from log.eval (used in log file naming)
            task_id: str | None = None
            eval_info = _obj_get(log, "eval") if log is not None else None
            if eval_info is not None:
                task_id = _obj_get(eval_info, "task_id")

            # Extract model name and log task scores
            if log is not None:
                model_name = _obj_get(eval_info, "model")
                if model_name:
                    self._task_models[eval_id].add(str(model_name))

                # Log task scores
                self._log_task_scores_client(client, run_id, log, task_name)

                # Aggregate model usage from stats
                stats = _obj_get(log, "stats")
                stats_usage = _obj_get(stats, "model_usage")
                if isinstance(stats_usage, dict):
                    # Reconcile totals at task end without additive accumulation.
                    self._set_usage_totals_for_task(eval_id, stats_usage)

            # Determine run status from TaskEnd log status.
            # TaskEnd does not include an exception field.
            log_status = str(_obj_get(log, "status") or "").lower()
            status = (
                "FINISHED" if (not log_status or log_status == "success") else "FAILED"
            )

            client.set_tag(run_id, f"{TAG_PREFIX}.status", status)

            # Log final summary metrics for this task
            total_samples = self._task_sample_counts.get(eval_id, 0)
            correct_samples = self._task_correct_counts.get(eval_id, 0)
            accuracy = correct_samples / total_samples if total_samples > 0 else 0.0

            client.log_metric(run_id, f"{TAG_PREFIX}.samples_total", total_samples)
            client.log_metric(run_id, f"{TAG_PREFIX}.samples_correct", correct_samples)
            client.log_metric(run_id, f"{TAG_PREFIX}.accuracy", accuracy)
            self._log_usage_metrics_client(client, run_id, eval_id)

            # Log artifacts for this task via client APIs (safe for parallel tasks)
            if cfg.log_artifacts:
                self._log_tables_for_task(client, run_id, eval_id)
                self._log_task_inspect_logs(
                    client,
                    run_id,
                    log,
                    eval_id=eval_id,
                    task_id=task_id,
                )

            # End the run for this task using client API
            with self._lock:
                mlflow_status = "FINISHED" if status == "FINISHED" else "FAILED"
                client.set_terminated(run_id, status=mlflow_status)
                del self._active_runs[eval_id]
                self._task_settings.pop(eval_id, None)

            _LOG.info(
                f"MLflow run completed for task {task_name} (eval_id={eval_id}): {run_id}"
            )

        except Exception:
            _LOG.exception("MLflow hook failed during task end")

    async def on_run_end(self, data: RunEnd) -> None:
        """Clean up after all tasks have completed.

        Note: Individual task runs are ended in on_task_end.
        This hook handles cleanup of autolog and run-level state.
        """
        if not self.enabled():
            return

        try:
            mlflow = self._mlflow()
            client = mlflow.tracking.MlflowClient()

            # End any orphaned runs (shouldn't happen normally)
            for eval_id, run_id in list(self._active_runs.items()):
                _LOG.warning(f"Orphaned run found for eval_id={eval_id}, ending it")
                try:
                    status = (
                        "FAILED" if getattr(data, "exception", None) else "FINISHED"
                    )
                    client.set_terminated(run_id, status=status)
                except Exception:
                    pass

            # Disable autolog
            if self._autolog_enabled:
                self._disable_autolog(mlflow)

            # Clean up run-level state (eval_set state is managed by on_eval_set_end)
            with self._lock:
                reset_run_state(self)

            _LOG.info("MLflow tracking cleanup completed")

        except Exception:
            _LOG.exception("MLflow hook failed during run end")

    # -------------------------------------------------------------------------
    # Autolog Management
    # -------------------------------------------------------------------------

    def _enable_autolog(self, mlflow: Any, models: list[str]) -> None:
        """Enable MLflow autolog for specified LLM libraries."""
        enabled_any = enable_autolog(
            models,
            find_spec=importlib.util.find_spec,
            import_module=importlib.import_module,
        )

        self._autolog_enabled = enabled_any
        if enabled_any:
            _LOG.info(f"MLflow autolog enabled for: {models}")

    def _disable_autolog(self, mlflow: Any) -> None:
        """Disable all autolog integrations."""
        try:
            mlflow.autolog(disable=True)
        except Exception:
            _LOG.debug("Could not disable autolog", exc_info=True)
        self._autolog_enabled = False

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    @staticmethod
    def _mlflow():
        """Lazy import of mlflow."""
        import mlflow

        return mlflow

    def _is_correct(self, sample: Any) -> bool:
        """Check if a sample is correct based on scores."""
        return is_correct(sample)

    def _ensure_experiment(self, mlflow: Any, name: str) -> None:
        """Create or get experiment by name."""
        ensure_experiment(mlflow, name)

    def _default_experiment_name(
        self, run_id: str | None, task_names: list[str] | None
    ) -> str:
        """Generate default experiment name."""
        return default_experiment_name(run_id, task_names)

    def _get_task_name(self, data: Any, log: Any) -> str | None:
        """Extract task name from TaskEnd data or log."""
        return get_task_name(data, log)

    def _get_sample_output_text(self, sample: Any) -> str | None:
        """Extract output text from sample."""
        return get_sample_output_text(sample)

    def _scores_to_dict(self, scores: Any) -> dict[str, Any]:
        """Convert scores to JSON-serializable dict."""
        return scores_to_dict(scores)

    @staticmethod
    def _rows_to_columns(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
        """Convert list of row dicts to column dict (for MLflow tables)."""
        return rows_to_columns(rows)
