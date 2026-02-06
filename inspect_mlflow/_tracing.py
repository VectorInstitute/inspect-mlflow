"""Tracing mixin for MLflow hooks."""

from __future__ import annotations

import logging
from typing import Any

from ._utils import (
    SPAN_TYPE_AGENT,
    SPAN_TYPE_CHAIN,
    SPAN_TYPE_LLM,
    SPAN_TYPE_TOOL,
    _clean_token,
    _obj_get,
    _to_json,
    _usage_to_dict,
)

_LOG = logging.getLogger(__name__)


class TracingMixin:
    """Mixin providing MLflow tracing methods for MLflowHooks.

    Expects the host class to provide:
    - self._trace_supported: bool
    - self._scores_to_dict(scores) -> dict
    - self._get_sample_output_text(sample) -> str | None
    """

    def _log_sample_trace(
        self,
        mlflow: Any,
        task_name: str,
        eval_id: str,
        sample: Any,
    ) -> None:
        """Create a hierarchical trace for a sample execution."""
        if not self._trace_supported:
            return

        sample_id = _obj_get(sample, "id")
        sample_input = _obj_get(sample, "input")
        sample_output = self._get_sample_output_text(sample)
        scores = _obj_get(sample, "scores")
        events = _obj_get(sample, "events")

        try:
            with mlflow.start_span(
                name=f"sample.{sample_id or 'unknown'}", span_type=SPAN_TYPE_AGENT
            ) as root_span:
                try:
                    mlflow.update_current_trace(
                        tags={
                            "inspect.task": str(task_name),
                            "inspect.eval_id": str(eval_id),
                            "inspect.sample_id": str(sample_id) if sample_id else "",
                        }
                    )
                except Exception:
                    pass

                root_span.set_attributes(
                    {
                        "inspect.task": task_name,
                        "inspect.eval_id": str(eval_id),
                        "inspect.sample_id": str(sample_id) if sample_id else "",
                    }
                )

                root_span.set_inputs({"input": _to_json(sample_input)})
                root_span.set_outputs(
                    {
                        "output": sample_output or "",
                        "scores": self._scores_to_dict(scores),
                    }
                )

                if isinstance(events, list):
                    self._log_event_spans(mlflow, events, task_name, eval_id, sample_id)

        except Exception:
            self._trace_supported = False
            _LOG.debug("MLflow tracing disabled for this run", exc_info=True)

    def _log_event_spans(
        self,
        mlflow: Any,
        events: list[Any],
        task_name: str,
        eval_id: str,
        sample_id: Any,
    ) -> None:
        """Create spans for model and tool events within a sample."""
        for idx, event in enumerate(events):
            event_type = _obj_get(event, "event") or _obj_get(event, "type")

            if event_type == "model":
                self._log_model_event_span(
                    mlflow, event, idx, task_name, eval_id, sample_id
                )
            elif event_type == "tool":
                self._log_tool_event_span(
                    mlflow, event, idx, task_name, eval_id, sample_id
                )
            elif event_type == "error":
                self._log_error_event_span(
                    mlflow, event, idx, task_name, eval_id, sample_id
                )

    def _log_model_event_span(
        self,
        mlflow: Any,
        event: Any,
        idx: int,
        task_name: str,
        eval_id: str,
        sample_id: Any,
    ) -> None:
        """Create a span for a model (LLM) call event."""
        model_name = _obj_get(event, "model") or "unknown"
        with mlflow.start_span(
            name=f"llm.{_clean_token(str(model_name), 32)}", span_type=SPAN_TYPE_LLM
        ) as span:
            span.set_attributes(
                {
                    "inspect.event_index": idx,
                    "model": str(model_name),
                }
            )

            input_data = _obj_get(event, "input")
            span.set_inputs({"messages": _to_json(input_data)})

            output = _obj_get(event, "output")
            completion = _obj_get(output, "completion")
            usage = _usage_to_dict(_obj_get(output, "usage"))

            span.set_outputs(
                {
                    "completion": _to_json(completion),
                    "usage": usage,
                }
            )

    def _log_tool_event_span(
        self,
        mlflow: Any,
        event: Any,
        idx: int,
        task_name: str,
        eval_id: str,
        sample_id: Any,
    ) -> None:
        """Create a span for a tool call event."""
        function_name = _obj_get(event, "function") or "unknown_tool"
        with mlflow.start_span(
            name=f"tool.{_clean_token(str(function_name), 32)}",
            span_type=SPAN_TYPE_TOOL,
        ) as span:
            span.set_attributes(
                {
                    "inspect.event_index": idx,
                    "tool.function": str(function_name),
                }
            )

            span.set_inputs(
                {
                    "function": function_name,
                    "arguments": _to_json(_obj_get(event, "arguments")),
                }
            )

            error = _obj_get(event, "error")
            span.set_outputs(
                {
                    "result": _to_json(_obj_get(event, "result")),
                    "error": _to_json(error) if error else None,
                }
            )

    def _log_error_event_span(
        self,
        mlflow: Any,
        event: Any,
        idx: int,
        task_name: str,
        eval_id: str,
        sample_id: Any,
    ) -> None:
        """Create a span for an error event."""
        with mlflow.start_span(name="error", span_type=SPAN_TYPE_CHAIN) as span:
            span.set_attributes({"inspect.event_index": idx})
            span.set_outputs({"error": _to_json(_obj_get(event, "error"))})
