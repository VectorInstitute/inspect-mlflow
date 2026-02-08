"""Tests for MLflowHooks."""

import logging
from unittest.mock import MagicMock

import pytest
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import CORRECT, INCORRECT, NOANSWER, PARTIAL
from inspect_mlflow._hook_helpers import get_sample_output_text
from inspect_mlflow._utils import (
    _clean_key,
    _clean_token,
    _coerce_metric,
    _iter_scores,
    _to_json,
    _usage_to_dict,
)
from inspect_mlflow.hooks import MLflowHooks


class TestHelperFunctions:
    """Test helper functions."""

    def test_clean_key(self):
        """Test key cleaning for MLflow."""
        assert _clean_key("hello world") == "hello_world"
        assert _clean_key("a/b/c") == "a.b.c"
        assert _clean_key("foo:bar") == "foo.bar"
        assert _clean_key("path\\to\\file") == "path.to.file"

    def test_clean_token(self):
        """Test token cleaning with truncation."""
        assert _clean_token("short") == "short"
        assert _clean_token("") == "empty"

        # Test truncation with hash
        long_value = "a" * 100
        result = _clean_token(long_value, max_len=48)
        assert len(result) <= 48
        assert "_" in result  # Should have hash suffix

    def test_coerce_metric_numbers(self):
        """Test coercing numeric values to metrics."""
        assert _coerce_metric(1) == 1.0
        assert _coerce_metric(0.5) == 0.5
        assert _coerce_metric(True) == 1.0
        assert _coerce_metric(False) == 0.0
        assert _coerce_metric(None) is None

    def test_coerce_metric_score_values(self):
        """Test coercing Inspect score constants."""
        assert _coerce_metric(CORRECT) == 1.0
        assert _coerce_metric(INCORRECT) == 0.0
        assert _coerce_metric(PARTIAL) == 0.5
        assert _coerce_metric(NOANSWER) == 0.0

    def test_coerce_metric_strings(self):
        """Test coercing string values."""
        assert _coerce_metric("yes") == 1.0
        assert _coerce_metric("true") == 1.0
        assert _coerce_metric("no") == 0.0
        assert _coerce_metric("false") == 0.0
        assert _coerce_metric("3.14") == 3.14
        assert _coerce_metric("invalid") is None

    def test_coerce_metric_with_value_attr(self):
        """Test coercing objects with .value attribute."""
        obj = MagicMock()
        obj.value = 0.75
        assert _coerce_metric(obj) == 0.75

    def test_to_json_primitives(self):
        """Test JSON conversion for primitives."""
        assert _to_json("hello") == "hello"
        assert _to_json(42) == "42"
        assert _to_json(3.14) == "3.14"
        assert _to_json(True) == "True"
        assert _to_json(None) is None

    def test_to_json_complex(self):
        """Test JSON conversion for complex objects."""
        assert _to_json({"a": 1}) == '{"a": 1}'
        assert _to_json([1, 2, 3]) == "[1, 2, 3]"

    def test_usage_to_dict(self):
        """Test extracting usage from various formats."""
        # Dict input
        usage_dict = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }
        result = _usage_to_dict(usage_dict)
        assert result == usage_dict

        # Object input
        usage_obj = MagicMock()
        usage_obj.input_tokens = 100
        usage_obj.output_tokens = 50
        usage_obj.total_tokens = 150
        result = _usage_to_dict(usage_obj)
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50

        # None input
        assert _usage_to_dict(None) == {}

    def test_iter_scores_dict(self):
        """Test iterating over dict scores."""
        scores = {"accuracy": 0.9, "f1": 0.85}
        result = list(_iter_scores(scores))
        assert ("accuracy", 0.9) in result
        assert ("f1", 0.85) in result

    def test_iter_scores_list(self):
        """Test iterating over list scores."""
        score1 = MagicMock(name="accuracy")
        score1.name = "accuracy"
        score2 = MagicMock(name="f1")
        score2.name = "f1"
        scores = [score1, score2]

        result = list(_iter_scores(scores))
        assert result[0][0] == "accuracy"
        assert result[1][0] == "f1"

    def test_get_sample_output_text_handles_empty_model_output(self):
        """ModelOutput.message can raise when choices are empty; helper must not."""
        sample = MagicMock()
        sample.output = ModelOutput(model="mockllm/model", choices=[], completion="")

        result = get_sample_output_text(sample)

        assert isinstance(result, str)

    def test_get_sample_output_text_prefers_completion_text(self):
        """Completion text should be used when present."""
        sample = MagicMock()
        sample.output = ModelOutput.from_content(
            model="mockllm/model", content="Hello from completion"
        )

        result = get_sample_output_text(sample)

        assert result == "Hello from completion"


class TestMLflowHooks:
    """Test MLflowHooks class."""

    def test_init(self, clean_env):
        """Test hook initialization."""
        hooks = MLflowHooks()

        assert hooks._active_runs == {}
        assert hooks._run_logging_enabled is False
        assert hooks._task_sample_counts == {}
        assert hooks._task_correct_counts == {}

    def test_enabled_default(self, clean_env):
        """Test enabled returns True by default."""
        hooks = MLflowHooks()
        assert hooks.enabled() is True

    def test_enabled_disabled_via_env(self, clean_env, monkeypatch):
        """Test enabled returns False when disabled."""
        monkeypatch.setenv("INSPECT_MLFLOW_ENABLED", "false")
        hooks = MLflowHooks()
        assert hooks.enabled() is False

    def test_is_correct(self, clean_env, sample_eval_sample):
        """Test _is_correct method."""
        hooks = MLflowHooks()

        # Correct sample
        sample_eval_sample.scores = {"acc": MagicMock(value=CORRECT)}
        assert hooks._is_correct(sample_eval_sample) is True

        sample_eval_sample.scores = {"acc": MagicMock(value=True)}
        assert hooks._is_correct(sample_eval_sample) is True

        sample_eval_sample.scores = {"acc": MagicMock(value=1)}
        assert hooks._is_correct(sample_eval_sample) is True

        sample_eval_sample.scores = {"acc": MagicMock(value=1.0)}
        assert hooks._is_correct(sample_eval_sample) is True

        # Partial scores should not count as correct.
        sample_eval_sample.scores = {"acc": MagicMock(value=PARTIAL)}
        assert hooks._is_correct(sample_eval_sample) is False

        # Incorrect sample
        sample_eval_sample.scores = {"acc": MagicMock(value=INCORRECT)}
        assert hooks._is_correct(sample_eval_sample) is False

        # No scores
        sample_eval_sample.scores = None
        assert hooks._is_correct(sample_eval_sample) is False

    def test_per_task_counters(self, clean_env):
        """Test per-task sample counters."""
        hooks = MLflowHooks()

        # Initialize counters
        hooks._task_sample_counts["eval1"] = 0
        hooks._task_correct_counts["eval1"] = 0

        # Increment
        hooks._task_sample_counts["eval1"] = 10
        hooks._task_correct_counts["eval1"] = 7

        accuracy = (
            hooks._task_correct_counts["eval1"] / hooks._task_sample_counts["eval1"]
        )
        assert accuracy == 0.7

        # Different eval_id gets independent counters
        hooks._task_sample_counts["eval2"] = 0
        assert hooks._task_sample_counts["eval2"] == 0
        assert hooks._task_sample_counts["eval1"] == 10

    def test_default_experiment_name(self, clean_env):
        """Test experiment name generation."""
        hooks = MLflowHooks()

        # With task names
        name = hooks._default_experiment_name("run123", ["task1", "task2"])
        assert name.startswith("inspect-task1-")

        # Without task names
        name = hooks._default_experiment_name("run123", None)
        assert name.startswith("inspect-eval-")

        # Without run id
        name = hooks._default_experiment_name(None, ["task1"])
        assert name == "inspect-task1"

    def test_scores_to_dict(self, clean_env):
        """Test converting scores to dict."""
        hooks = MLflowHooks()

        scores = {
            "accuracy": MagicMock(value=0.9),
            "f1": MagicMock(value=0.85),
        }

        result = hooks._scores_to_dict(scores)
        assert "accuracy" in result
        assert "f1" in result

    def test_log_sample_trace_emits_message_spans(self, clean_env):
        """Conversation messages should be visible in MLflow trace spans."""
        hooks = MLflowHooks()

        class FakeSpan:
            def __init__(
                self,
                owner: "FakeMlflow",
                name: str,
                span_type: str,
                *,
                trace_id: str = "trace-1",
                trace_destination: object | None = None,
            ) -> None:
                self.owner = owner
                self.name = name
                self.span_type = span_type
                self.trace_id = trace_id
                self.trace_destination = trace_destination
                self.attributes: dict[str, object] = {}
                self.inputs: dict[str, object] = {}
                self.outputs: dict[str, object] = {}

            def __enter__(self) -> "FakeSpan":
                self.owner.spans.append(self)
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def set_attributes(self, attrs: dict[str, object]) -> None:
                self.attributes.update(attrs)

            def set_inputs(self, inputs: dict[str, object]) -> None:
                self.inputs.update(inputs)

            def set_outputs(self, outputs: dict[str, object]) -> None:
                self.outputs.update(outputs)

        class FakeMlflow:
            def __init__(self) -> None:
                self.spans: list[FakeSpan] = []
                self.trace_tags: dict[str, str] = {}

            def start_span(
                self,
                name: str,
                span_type: str,
                trace_destination: object | None = None,
            ) -> FakeSpan:
                return FakeSpan(
                    self,
                    name,
                    span_type,
                    trace_destination=trace_destination,
                )

            def update_current_trace(self, tags: dict[str, str]) -> None:
                self.trace_tags.update(tags)

        sample = MagicMock()
        sample.id = "sample-1"
        sample.input = "What is 2 + 2?"
        sample.output = ModelOutput.from_content(
            model="openai/gpt-4o-mini", content="The answer is 4."
        )
        sample.scores = {"accuracy": MagicMock(value=1)}
        sample.messages = [
            {"role": "user", "content": "What is 2 + 2?"},
            {
                "role": "assistant",
                "content": "The answer is 4.",
                "model": "openai/gpt-4o-mini",
            },
        ]
        sample.events = []

        mlflow = FakeMlflow()
        trace_id = hooks._log_sample_trace(
            mlflow=mlflow, task_name="task-a", eval_id="eval-1", sample=sample
        )

        assert trace_id == "trace-1"

        span_by_name = {span.name: span for span in mlflow.spans}
        assert "sample.sample-1" in span_by_name
        assert "message.000.user" in span_by_name
        assert "message.001.assistant" in span_by_name

        root_span = span_by_name["sample.sample-1"]
        assert root_span.inputs["messages_count"] == 2

        user_span = span_by_name["message.000.user"]
        assert user_span.inputs["content"] == "What is 2 + 2?"

        assistant_span = span_by_name["message.001.assistant"]
        assert assistant_span.outputs["content"] == "The answer is 4."

    def test_enable_autolog_uses_supported_kwargs_only(self, clean_env, monkeypatch):
        """Autolog should work when flavor autolog doesn't accept log_models."""
        hooks = MLflowHooks()
        called: dict[str, bool] = {}

        def fake_autolog(*, log_traces: bool = True) -> None:
            called["log_traces"] = log_traces

        fake_module = MagicMock()
        fake_module.autolog = fake_autolog

        monkeypatch.setattr(
            "inspect_mlflow.hooks.importlib.util.find_spec", lambda _name: object()
        )
        monkeypatch.setattr(
            "inspect_mlflow.hooks.importlib.import_module",
            lambda _module_name: fake_module,
        )

        hooks._enable_autolog(MagicMock(), ["openai"])

        assert called["log_traces"] is True
        assert hooks._autolog_enabled is True

    def test_rows_to_columns(self, clean_env):
        """Test converting rows to column format."""
        rows = [
            {"a": 1, "b": 2},
            {"a": 3, "b": 4},
            {"a": 5, "b": 6},
        ]

        result = MLflowHooks._rows_to_columns(rows)

        assert result == {"a": [1, 3, 5], "b": [2, 4, 6]}

    def test_rows_to_columns_missing_keys(self, clean_env):
        """Test rows_to_columns handles missing keys."""
        rows = [
            {"a": 1, "b": 2},
            {"a": 3},  # Missing 'b'
            {"a": 5, "b": 6, "c": 7},  # Extra 'c'
        ]

        result = MLflowHooks._rows_to_columns(rows)

        assert result["a"] == [1, 3, 5]
        assert result["b"] == [2, None, 6]
        assert result["c"] == [None, None, 7]

    def test_log_tables_for_task_includes_messages_table(self, clean_env):
        """Messages table should be written as an MLflow artifact."""
        hooks = MLflowHooks()
        client = MagicMock()
        eval_id = "eval-1"

        hooks._task_message_rows[eval_id].append(
            {
                "task_name": "task-a",
                "eval_id": eval_id,
                "sample_id": "s1",
                "message_index": 0,
                "role": "user",
                "content": "hello",
            }
        )

        hooks._log_tables_for_task(client, "run-1", eval_id)

        artifact_files = [
            call.kwargs.get("artifact_file") for call in client.log_table.call_args_list
        ]
        assert "inspect/messages.json" in artifact_files

    def test_log_task_inspect_logs_dedupes_same_file(self, clean_env, tmp_path):
        """Avoid uploading the same log twice when URI and glob find same file."""
        hooks = MLflowHooks()
        client = MagicMock()
        task_id = "abc123"
        log_file = tmp_path / f"2026-01-01_task_{task_id}.json"
        log_file.write_text("{}")

        eval_info = MagicMock()
        eval_info.run_id = "inspect-run-1"
        eval_info.task_id = task_id

        log = MagicMock()
        log.location = log_file.as_uri()
        log.eval = eval_info

        hooks._log_task_inspect_logs(
            client,
            run_id="mlflow-run-1",
            log=log,
            task_id=task_id,
        )

        assert client.log_artifact.call_count == 1
        kwargs = client.log_artifact.call_args.kwargs
        assert kwargs["run_id"] == "mlflow-run-1"
        assert kwargs["local_path"] == str(log_file)
        assert kwargs["artifact_path"] == "inspect/logs"

    def test_log_task_inspect_logs_prefers_explicit_location_over_glob_matches(
        self, clean_env, tmp_path
    ):
        """Do not upload stale retry logs when exact location is available."""
        hooks = MLflowHooks()
        client = MagicMock()
        task_id = "abc123"
        old_log = tmp_path / f"2026-01-01_task_{task_id}.json"
        new_log = tmp_path / f"2026-01-02_task_{task_id}.json"
        old_log.write_text("{}")
        new_log.write_text("{}")

        eval_info = MagicMock()
        eval_info.run_id = "inspect-run-1"
        eval_info.task_id = task_id

        log = MagicMock()
        log.location = new_log.as_uri()
        log.eval = eval_info

        hooks._log_task_inspect_logs(
            client,
            run_id="mlflow-run-1",
            log=log,
            task_id=task_id,
        )

        assert client.log_artifact.call_count == 1
        kwargs = client.log_artifact.call_args.kwargs
        assert kwargs["local_path"] == str(new_log)


class TestMLflowHooksAsync:
    """Test async hook methods."""

    @pytest.mark.asyncio
    async def test_on_run_start_disabled(self, clean_env, monkeypatch):
        """Test on_run_start does nothing when disabled."""
        monkeypatch.setenv("INSPECT_MLFLOW_ENABLED", "false")
        hooks = MLflowHooks()

        data = MagicMock()
        await hooks.on_run_start(data)

        assert hooks._run_logging_enabled is False

    @pytest.mark.asyncio
    async def test_on_sample_end_disabled(self, clean_env, monkeypatch):
        """Test on_sample_end does nothing when disabled."""
        monkeypatch.setenv("INSPECT_MLFLOW_ENABLED", "false")
        hooks = MLflowHooks()

        data = MagicMock()
        await hooks.on_sample_end(data)

        assert hooks._task_sample_counts == {}

    @pytest.mark.asyncio
    async def test_on_sample_end_not_logging(self, clean_env):
        """Test on_sample_end does nothing when not logging."""
        hooks = MLflowHooks()
        hooks._run_logging_enabled = False

        data = MagicMock()
        await hooks.on_sample_end(data)

        assert hooks._task_sample_counts == {}

    @pytest.mark.asyncio
    async def test_reuses_single_mlflow_client_within_run(self, clean_env, monkeypatch):
        """Hooks should reuse one MlflowClient per run to avoid connection churn."""
        monkeypatch.setenv("INSPECT_MLFLOW_LOG_ARTIFACTS", "false")
        monkeypatch.setenv("INSPECT_MLFLOW_LOG_TRACES", "false")
        monkeypatch.setenv("INSPECT_MLFLOW_AUTOLOG_ENABLED", "false")

        hooks = MLflowHooks()

        client = MagicMock()
        experiment = MagicMock()
        experiment.experiment_id = "exp-1"
        run = MagicMock()
        run.info.run_id = "run-1"
        client.get_experiment_by_name.return_value = experiment
        client.create_run.return_value = run

        mlflow = MagicMock()
        mlflow.tracking.MlflowClient.return_value = client
        hooks._mlflow = lambda: mlflow  # type: ignore[assignment]

        run_start = MagicMock()
        run_start.run_id = "inspect-run-1"
        run_start.task_names = ["task-a"]
        run_start.eval_set_id = None
        await hooks.on_run_start(run_start)

        spec = MagicMock()
        spec.task = "task-a"
        spec.name = "task-a"
        spec.model = "openai/gpt-4o-mini"
        spec.metadata = None
        spec.model_dump.return_value = {
            "task": "task-a",
            "model": "openai/gpt-4o-mini",
            "dataset": {"name": "test"},
            "config": {},
        }

        task_start = MagicMock()
        task_start.eval_id = "eval-1"
        task_start.spec = spec
        await hooks.on_task_start(task_start)

        sample = MagicMock()
        sample.id = "sample-1"
        sample.input = "prompt"
        sample.target = "target"
        sample.output = ModelOutput.from_content(
            model="openai/gpt-4o-mini", content="done"
        )
        sample.scores = None
        sample.model_usage = {}
        sample.events = []
        sample.total_time = 0.0
        sample.working_time = 0.0
        sample.error = None

        sample_end = MagicMock()
        sample_end.eval_id = "eval-1"
        sample_end.sample = sample
        await hooks.on_sample_end(sample_end)

        eval_info = MagicMock()
        eval_info.task = "task-a"
        eval_info.task_id = "task-a"
        eval_info.model = "openai/gpt-4o-mini"

        log = MagicMock()
        log.status = "success"
        log.eval = eval_info
        log.scores = None
        log.metrics = None
        log.results = None
        log.stats = None

        task_end = MagicMock()
        task_end.eval_id = "eval-1"
        task_end.log = log
        task_end.spec = None
        await hooks.on_task_end(task_end)

        assert mlflow.tracking.MlflowClient.call_count == 1

    @pytest.mark.asyncio
    async def test_on_sample_end_handles_empty_model_output_choices(
        self, clean_env, monkeypatch
    ):
        """Cancelled/empty outputs should not crash sample logging."""
        monkeypatch.setenv("INSPECT_MLFLOW_LOG_TRACES", "false")
        hooks = MLflowHooks()
        hooks._run_logging_enabled = True
        hooks._active_runs["eval-1"] = "run-1"

        client = MagicMock()
        mlflow = MagicMock()
        mlflow.tracking.MlflowClient.return_value = client
        hooks._mlflow = lambda: mlflow  # type: ignore[assignment]

        sample = MagicMock()
        sample.id = "sample-1"
        sample.input = "prompt"
        sample.target = "target"
        sample.output = ModelOutput(
            model="openai/gpt-4o-mini", choices=[], completion=""
        )
        sample.scores = None
        sample.events = []
        sample.model_usage = {}
        sample.total_time = 0.0
        sample.working_time = 0.0
        sample.error = None
        sample.messages = [
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": "reply"},
        ]

        data = MagicMock()
        data.eval_id = "eval-1"
        data.sample = sample

        await hooks.on_sample_end(data)

        assert hooks._task_sample_counts["eval-1"] == 1
        assert len(hooks._task_sample_rows["eval-1"]) == 1
        assert len(hooks._task_message_rows["eval-1"]) == 2
        assert hooks._task_message_rows["eval-1"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_on_task_end_logs_zero_accuracy_for_unscored_tasks(
        self, clean_env, monkeypatch
    ):
        """Compatibility: emit accuracy/samples_correct as 0 when unscored."""
        monkeypatch.setenv("INSPECT_MLFLOW_LOG_ARTIFACTS", "false")

        hooks = MLflowHooks()
        hooks._run_logging_enabled = True
        hooks._active_runs["eval-1"] = "run-1"
        hooks._task_sample_counts["eval-1"] = 1
        hooks._task_scored_counts["eval-1"] = 0
        hooks._task_correct_counts["eval-1"] = 0

        client = MagicMock()
        mlflow = MagicMock()
        mlflow.tracking.MlflowClient.return_value = client
        hooks._mlflow = lambda: mlflow  # type: ignore[assignment]

        eval_info = MagicMock()
        eval_info.task = "task-a"
        eval_info.task_id = "task-a"
        eval_info.model = "openai/gpt-4o-mini"

        log = MagicMock()
        log.status = "success"
        log.eval = eval_info
        log.scores = None
        log.metrics = None
        log.results = None
        log.stats = None

        data = MagicMock()
        data.eval_id = "eval-1"
        data.log = log
        data.spec = None

        await hooks.on_task_end(data)

        client.log_metric.assert_any_call("run-1", "inspect.samples_total", 1)
        client.log_metric.assert_any_call("run-1", "inspect.samples_scored", 0)
        client.log_metric.assert_any_call("run-1", "inspect.samples_correct", 0)
        client.log_metric.assert_any_call("run-1", "inspect.accuracy", 0.0)

    @pytest.mark.asyncio
    async def test_usage_falls_back_to_model_events_when_sample_usage_empty(
        self, clean_env, monkeypatch
    ):
        """Token metrics should still log from event output usage when available."""
        monkeypatch.setenv("INSPECT_MLFLOW_LOG_ARTIFACTS", "false")
        monkeypatch.setenv("INSPECT_MLFLOW_LOG_TRACES", "false")

        hooks = MLflowHooks()
        hooks._run_logging_enabled = True
        hooks._active_runs["eval-1"] = "run-1"

        client = MagicMock()
        mlflow = MagicMock()
        mlflow.tracking.MlflowClient.return_value = client
        hooks._mlflow = lambda: mlflow  # type: ignore[assignment]

        sample = MagicMock()
        sample.id = "sample-1"
        sample.input = "prompt"
        sample.target = "target"
        sample.output = ModelOutput.from_content(
            model="openai/gpt-4o-mini", content="done"
        )
        sample.scores = None
        sample.model_usage = {}
        sample.events = [
            {
                "event": "model",
                "model": "openai/gpt-4o-mini",
                "output": {
                    "completion": "done",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                    },
                },
            }
        ]
        sample.total_time = 0.0
        sample.working_time = 0.0
        sample.error = None

        sample_data = MagicMock()
        sample_data.eval_id = "eval-1"
        sample_data.sample = sample

        await hooks.on_sample_end(sample_data)

        eval_info = MagicMock()
        eval_info.task = "task-a"
        eval_info.task_id = "task-a"
        eval_info.model = "openai/gpt-4o-mini"

        stats = MagicMock()
        stats.model_usage = {}

        log = MagicMock()
        log.status = "success"
        log.eval = eval_info
        log.scores = None
        log.metrics = None
        log.results = None
        log.stats = stats

        end_data = MagicMock()
        end_data.eval_id = "eval-1"
        end_data.log = log
        end_data.spec = None

        await hooks.on_task_end(end_data)

        clean_model = _clean_token("openai/gpt-4o-mini", max_len=64)
        client.log_metric.assert_any_call(
            "run-1",
            _clean_key(f"inspect.usage.{clean_model}.input_tokens"),
            10.0,
        )
        client.log_metric.assert_any_call(
            "run-1",
            _clean_key(f"inspect.usage.{clean_model}.output_tokens"),
            5.0,
        )
        client.log_metric.assert_any_call(
            "run-1",
            _clean_key("inspect.tokens.total_tokens"),
            15.0,
        )

    @pytest.mark.asyncio
    async def test_on_task_end_uses_log_status_for_failure(
        self, clean_env, monkeypatch
    ):
        """Test on_task_end marks run failed when task log status is error."""
        monkeypatch.setenv("INSPECT_MLFLOW_LOG_ARTIFACTS", "false")
        hooks = MLflowHooks()
        hooks._run_logging_enabled = True
        hooks._active_runs["eval-1"] = "run-1"

        client = MagicMock()
        mlflow = MagicMock()
        mlflow.tracking.MlflowClient.return_value = client
        hooks._mlflow = lambda: mlflow  # type: ignore[assignment]

        eval_info = MagicMock()
        eval_info.task = "task-a"
        eval_info.task_id = "task-a"
        eval_info.model = "openai/gpt-4o-mini"

        log = MagicMock()
        log.status = "error"
        log.eval = eval_info
        log.scores = None
        log.metrics = None
        log.results = None
        log.stats = None

        data = MagicMock()
        data.eval_id = "eval-1"
        data.log = log
        data.spec = None

        await hooks.on_task_end(data)

        client.set_tag.assert_any_call("run-1", "inspect.status", "FAILED")
        client.set_terminated.assert_called_once_with("run-1", status="FAILED")
        assert "eval-1" not in hooks._active_runs

    @pytest.mark.asyncio
    async def test_task_metadata_disable_does_not_leak_globally(self, clean_env):
        """Task metadata overrides should not mutate global hook settings."""
        hooks = MLflowHooks()
        hooks._run_logging_enabled = True
        hooks._experiment_name = "inspect-exp"

        client = MagicMock()
        experiment = MagicMock()
        experiment.experiment_id = "exp-1"
        run = MagicMock()
        run.info.run_id = "run-2"
        client.get_experiment_by_name.return_value = experiment
        client.create_run.return_value = run

        mlflow = MagicMock()
        mlflow.tracking.MlflowClient.return_value = client
        hooks._mlflow = lambda: mlflow  # type: ignore[assignment]

        # Task A disables logging via metadata and should not affect global settings.
        spec_disabled = MagicMock()
        spec_disabled.task = "task-disabled"
        spec_disabled.name = "task-disabled"
        spec_disabled.model = "openai/gpt-4o-mini"
        spec_disabled.metadata = {"inspect_mlflow_enabled": False}

        data_disabled = MagicMock()
        data_disabled.eval_id = "eval-disabled"
        data_disabled.spec = spec_disabled

        await hooks.on_task_start(data_disabled)

        assert hooks.enabled() is True
        assert "eval-disabled" not in hooks._active_runs

        # Task B with no metadata should still create an MLflow run.
        spec_enabled = MagicMock()
        spec_enabled.task = "task-enabled"
        spec_enabled.name = "task-enabled"
        spec_enabled.model = "openai/gpt-4o-mini"
        spec_enabled.metadata = None

        data_enabled = MagicMock()
        data_enabled.eval_id = "eval-enabled"
        data_enabled.spec = spec_enabled

        await hooks.on_task_start(data_enabled)

        assert hooks._active_runs["eval-enabled"] == "run-2"

    @pytest.mark.asyncio
    async def test_task_metadata_disable_skips_no_active_run_warning(
        self, clean_env, caplog
    ):
        """Task-end should not warn when task was explicitly disabled via metadata."""
        hooks = MLflowHooks()
        hooks._run_logging_enabled = True

        spec_disabled = MagicMock()
        spec_disabled.task = "task-disabled"
        spec_disabled.name = "task-disabled"
        spec_disabled.model = "openai/gpt-4o-mini"
        spec_disabled.metadata = {"inspect_mlflow_enabled": False}

        data_disabled = MagicMock()
        data_disabled.eval_id = "eval-disabled"
        data_disabled.spec = spec_disabled

        await hooks.on_task_start(data_disabled)
        assert "eval-disabled" in hooks._task_disabled_eval_ids

        end_data = MagicMock()
        end_data.eval_id = "eval-disabled"
        end_data.log = MagicMock(status="success")

        with caplog.at_level(logging.WARNING, logger="inspect_mlflow.hooks"):
            await hooks.on_task_end(end_data)

        assert "No active run found for eval_id=eval-disabled" not in caplog.text
        assert "eval-disabled" not in hooks._task_disabled_eval_ids

    @pytest.mark.asyncio
    async def test_task_end_usage_stats_do_not_double_count(
        self, clean_env, monkeypatch
    ):
        """Task-end stats usage should not be added on top of sample totals."""
        monkeypatch.setenv("INSPECT_MLFLOW_LOG_ARTIFACTS", "false")

        hooks = MLflowHooks()
        hooks._run_logging_enabled = True
        hooks._active_runs["eval-1"] = "run-1"
        hooks._task_usage_totals["eval-1"] = {
            "openai/gpt-4o-mini": {"input_tokens": 100, "output_tokens": 50}
        }

        client = MagicMock()
        mlflow = MagicMock()
        mlflow.tracking.MlflowClient.return_value = client
        hooks._mlflow = lambda: mlflow  # type: ignore[assignment]

        eval_info = MagicMock()
        eval_info.task = "task-a"
        eval_info.task_id = "task-a"
        eval_info.model = "openai/gpt-4o-mini"

        stats = MagicMock()
        stats.model_usage = {
            "openai/gpt-4o-mini": {"input_tokens": 100, "output_tokens": 50}
        }

        log = MagicMock()
        log.status = "success"
        log.eval = eval_info
        log.scores = None
        log.metrics = None
        log.results = None
        log.stats = stats

        data = MagicMock()
        data.eval_id = "eval-1"
        data.log = log
        data.spec = None

        await hooks.on_task_end(data)

        clean_model = _clean_token("openai/gpt-4o-mini", max_len=64)
        client.log_metric.assert_any_call(
            "run-1",
            _clean_key(f"inspect.usage.{clean_model}.input_tokens"),
            100.0,
        )
        client.log_metric.assert_any_call(
            "run-1",
            _clean_key(f"inspect.usage.{clean_model}.output_tokens"),
            50.0,
        )
        client.log_metric.assert_any_call(
            "run-1",
            _clean_key("inspect.tokens.input_tokens"),
            100.0,
        )
        client.log_metric.assert_any_call(
            "run-1",
            _clean_key("inspect.tokens.output_tokens"),
            50.0,
        )
        assert "eval-1" not in hooks._task_usage_totals

    @pytest.mark.asyncio
    async def test_task_end_reconciles_summary_from_log_samples_on_retry_resume(
        self, clean_env, monkeypatch
    ):
        """Task-end summary should include reused samples from task log."""
        monkeypatch.setenv("INSPECT_MLFLOW_LOG_ARTIFACTS", "false")

        hooks = MLflowHooks()
        hooks._run_logging_enabled = True
        hooks._active_runs["eval-1"] = "run-1"
        hooks._task_sample_counts["eval-1"] = 1
        hooks._task_scored_counts["eval-1"] = 1
        hooks._task_correct_counts["eval-1"] = 1
        hooks._task_scorer_names_by_eval_id["eval-1"] = ["accuracy"]

        client = MagicMock()
        mlflow = MagicMock()
        mlflow.tracking.MlflowClient.return_value = client
        hooks._mlflow = lambda: mlflow  # type: ignore[assignment]

        eval_info = MagicMock()
        eval_info.task = "task-a"
        eval_info.task_id = "task-a"
        eval_info.model = "openai/gpt-4o-mini"

        sample_1 = MagicMock()
        sample_1.scores = {"accuracy": MagicMock(value=1)}
        sample_2 = MagicMock()
        sample_2.scores = {"accuracy": MagicMock(value=0)}
        sample_3 = MagicMock()
        sample_3.scores = {"accuracy": MagicMock(value=1)}

        results = MagicMock()
        results.total_samples = 3

        log = MagicMock()
        log.status = "success"
        log.eval = eval_info
        log.scores = None
        log.metrics = None
        log.results = results
        log.stats = None
        log.samples = [sample_1, sample_2, sample_3]

        data = MagicMock()
        data.eval_id = "eval-1"
        data.log = log
        data.spec = None

        await hooks.on_task_end(data)

        metric_calls = [call.args for call in client.log_metric.call_args_list]
        assert ("run-1", "inspect.samples_total", 3) in metric_calls
        assert ("run-1", "inspect.samples_scored", 3) in metric_calls
        assert ("run-1", "inspect.samples_correct", 2) in metric_calls
        assert any(
            call[0] == "run-1"
            and call[1] == "inspect.accuracy"
            and call[2] == pytest.approx(2.0 / 3.0)
            for call in metric_calls
        )

    @pytest.mark.asyncio
    async def test_on_task_end_cleans_all_per_task_state(self, clean_env, monkeypatch):
        """Task end should release per-eval buffers to avoid memory growth."""
        monkeypatch.setenv("INSPECT_MLFLOW_LOG_ARTIFACTS", "false")

        hooks = MLflowHooks()
        hooks._run_logging_enabled = True
        hooks._active_runs["eval-1"] = "run-1"
        hooks._task_settings["eval-1"] = hooks.settings
        hooks._task_names_by_eval_id["eval-1"] = "task-a"
        hooks._task_sample_counts["eval-1"] = 1
        hooks._task_correct_counts["eval-1"] = 1
        hooks._task_sample_steps["eval-1"] = 1
        hooks._task_models["eval-1"].add("openai/gpt-4o-mini")
        hooks._task_raw_scores["eval-1"][("task-a", "score")]["1"] = 1
        hooks._task_usage_totals["eval-1"] = {"openai/gpt-4o-mini": {"input_tokens": 1}}
        hooks._task_experiment_ids["eval-1"] = "exp-1"
        hooks._task_sample_rows["eval-1"].append({"sample_id": "s1"})
        hooks._task_message_rows["eval-1"].append({"message_index": 0})
        hooks._task_sample_score_rows["eval-1"].append({"scorer": "acc"})
        hooks._task_rows_data["eval-1"].append({"task_name": "task-a"})
        hooks._task_event_rows["eval-1"].append({"event": "model"})
        hooks._task_usage_rows["eval-1"].append({"key": "input_tokens"})

        client = MagicMock()
        mlflow = MagicMock()
        mlflow.tracking.MlflowClient.return_value = client
        hooks._mlflow = lambda: mlflow  # type: ignore[assignment]

        eval_info = MagicMock()
        eval_info.task = "task-a"
        eval_info.task_id = "task-a"
        eval_info.model = "openai/gpt-4o-mini"

        log = MagicMock()
        log.status = "success"
        log.eval = eval_info
        log.scores = None
        log.metrics = None
        log.results = None
        log.stats = None

        data = MagicMock()
        data.eval_id = "eval-1"
        data.log = log
        data.spec = None

        await hooks.on_task_end(data)

        assert "eval-1" not in hooks._active_runs
        assert "eval-1" not in hooks._task_settings
        assert "eval-1" not in hooks._task_names_by_eval_id
        assert "eval-1" not in hooks._task_sample_counts
        assert "eval-1" not in hooks._task_correct_counts
        assert "eval-1" not in hooks._task_sample_steps
        assert "eval-1" not in hooks._task_models
        assert "eval-1" not in hooks._task_raw_scores
        assert "eval-1" not in hooks._task_usage_totals
        assert "eval-1" not in hooks._task_experiment_ids
        assert "eval-1" not in hooks._task_sample_rows
        assert "eval-1" not in hooks._task_message_rows
        assert "eval-1" not in hooks._task_sample_score_rows
        assert "eval-1" not in hooks._task_rows_data
        assert "eval-1" not in hooks._task_event_rows
        assert "eval-1" not in hooks._task_usage_rows
