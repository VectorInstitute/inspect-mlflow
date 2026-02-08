"""Tracing mixin for MLflow hooks."""

from __future__ import annotations

import logging
from typing import Any, Protocol

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

# Guardrails to keep traces small enough for MLflow UI rendering.
TRACE_MAX_EVENT_SPANS = 400
TRACE_MAX_MESSAGE_SPANS = 200
TRACE_MAX_TEXT_CHARS = 4000


class _TracingHost(Protocol):
    """Protocol describing hook attributes/methods used by TracingMixin."""

    def _scores_to_dict(self, scores: Any) -> dict[str, Any]: ...

    def _get_sample_output_text(self, sample: Any) -> str | None: ...

    def _log_event_spans(
        self,
        mlflow: Any,
        events: list[Any],
        task_name: str,
        eval_id: str,
        sample_id: Any,
    ) -> None: ...

    def _log_message_spans(
        self,
        mlflow: Any,
        messages: list[Any],
        task_name: str,
        eval_id: str,
        sample_id: Any,
    ) -> None: ...

    def _log_model_event_span(
        self,
        mlflow: Any,
        event: Any,
        idx: int,
        task_name: str,
        eval_id: str,
        sample_id: Any,
    ) -> None: ...

    def _log_tool_event_span(
        self,
        mlflow: Any,
        event: Any,
        idx: int,
        task_name: str,
        eval_id: str,
        sample_id: Any,
    ) -> None: ...

    def _log_error_event_span(
        self,
        mlflow: Any,
        event: Any,
        idx: int,
        task_name: str,
        eval_id: str,
        sample_id: Any,
    ) -> None: ...


class TracingMixin:
    """Mixin providing MLflow tracing methods for MLflowHooks.

    Expects the host class to provide:
    - self._scores_to_dict(scores) -> dict
    - self._get_sample_output_text(sample) -> str | None
    """

    @staticmethod
    def _truncate_trace_text(value: Any, max_chars: int = TRACE_MAX_TEXT_CHARS) -> Any:
        """Limit very large text payloads so trace retrieval/UI remains usable."""
        if not isinstance(value, str):
            return value
        if len(value) <= max_chars:
            return value
        truncated_chars = len(value) - max_chars
        return f"{value[:max_chars]}...[truncated {truncated_chars} chars]"

    def _log_sample_trace(
        self: _TracingHost,
        mlflow: Any,
        task_name: str,
        eval_id: str,
        sample: Any,
        experiment_id: str | None = None,
    ) -> str | None:
        """Create a hierarchical trace for a sample execution.

        Returns the generated trace_id when available.
        """
        sample_id = _obj_get(sample, "id")
        sample_input = _obj_get(sample, "input")
        sample_output = self._get_sample_output_text(sample)
        scores = _obj_get(sample, "scores")
        events = _obj_get(sample, "events")
        messages = _obj_get(sample, "messages")
        trace_id: str | None = None

        try:
            trace_destination: Any | None = None
            if experiment_id:
                try:
                    from mlflow.entities.trace_location import (
                        MlflowExperimentLocation,
                    )

                    trace_destination = MlflowExperimentLocation(experiment_id)
                except Exception:
                    trace_destination = None

            with mlflow.start_span(
                name=f"sample.{sample_id or 'unknown'}",
                span_type=SPAN_TYPE_AGENT,
                trace_destination=trace_destination,
            ) as root_span:
                trace_id = str(getattr(root_span, "trace_id", "") or "") or None
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

                root_inputs: dict[str, Any] = {
                    "input": TracingMixin._truncate_trace_text(_to_json(sample_input))
                }
                if isinstance(messages, list):
                    root_inputs["messages_count"] = len(messages)
                root_span.set_inputs(root_inputs)
                root_span.set_outputs(
                    {
                        "output": TracingMixin._truncate_trace_text(
                            sample_output or ""
                        ),
                        "scores": self._scores_to_dict(scores),
                    }
                )

                if isinstance(messages, list):
                    self._log_message_spans(
                        mlflow, messages, task_name, eval_id, sample_id
                    )

                if isinstance(events, list):
                    self._log_event_spans(mlflow, events, task_name, eval_id, sample_id)

            return trace_id

        except Exception:
            _LOG.debug("Could not log sample trace", exc_info=True)
            return None

    def _log_event_spans(
        self: _TracingHost,
        mlflow: Any,
        events: list[Any],
        task_name: str,
        eval_id: str,
        sample_id: Any,
    ) -> None:
        """Create spans for model and tool events within a sample."""
        for idx, event in enumerate(events[:TRACE_MAX_EVENT_SPANS]):
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

        if len(events) > TRACE_MAX_EVENT_SPANS:
            with mlflow.start_span(
                name="events.truncated",
                span_type=SPAN_TYPE_CHAIN,
            ) as span:
                span.set_attributes(
                    {
                        "inspect.event_count_total": len(events),
                        "inspect.event_count_logged": TRACE_MAX_EVENT_SPANS,
                        "inspect.task": task_name,
                        "inspect.eval_id": str(eval_id),
                        "inspect.sample_id": str(sample_id) if sample_id else "",
                    }
                )
                span.set_outputs(
                    {
                        "note": (
                            f"Truncated event spans from {len(events)} to "
                            f"{TRACE_MAX_EVENT_SPANS} to keep trace size manageable."
                        )
                    }
                )

    def _log_message_spans(
        self: _TracingHost,
        mlflow: Any,
        messages: list[Any],
        task_name: str,
        eval_id: str,
        sample_id: Any,
    ) -> None:
        """Create spans for conversation messages so they are easy to inspect."""
        for idx, message in enumerate(messages[:TRACE_MAX_MESSAGE_SPANS]):
            role_obj = _obj_get(message, "role")
            role_value = _obj_get(role_obj, "value")
            role = str(role_value if role_value is not None else role_obj or "unknown")
            role_lower = role.lower()

            content = TracingMixin._truncate_trace_text(
                _to_json(_obj_get(message, "content"))
            )
            source = _obj_get(message, "source")
            model = _obj_get(message, "model")
            stop_reason = _obj_get(message, "stop_reason")
            tool_calls = TracingMixin._truncate_trace_text(
                _to_json(_obj_get(message, "tool_calls"))
            )
            tool_call_id = TracingMixin._truncate_trace_text(
                _to_json(_obj_get(message, "tool_call_id"))
            )

            with mlflow.start_span(
                name=f"message.{idx:03d}.{_clean_token(role, 24)}",
                span_type=SPAN_TYPE_CHAIN,
            ) as span:
                attrs: dict[str, Any] = {
                    "inspect.message_index": idx,
                    "inspect.message_role": role,
                    "inspect.task": task_name,
                    "inspect.eval_id": str(eval_id),
                    "inspect.sample_id": str(sample_id) if sample_id else "",
                }
                if source is not None:
                    attrs["inspect.message_source"] = str(source)
                if model is not None:
                    attrs["inspect.message_model"] = str(model)
                span.set_attributes(attrs)

                payload = {
                    "content": content,
                    "tool_calls": tool_calls,
                    "tool_call_id": tool_call_id,
                    "stop_reason": TracingMixin._truncate_trace_text(
                        _to_json(stop_reason)
                    ),
                }
                if role_lower in {"assistant", "tool"}:
                    span.set_outputs(payload)
                else:
                    span.set_inputs(payload)

        if len(messages) > TRACE_MAX_MESSAGE_SPANS:
            with mlflow.start_span(
                name="messages.truncated",
                span_type=SPAN_TYPE_CHAIN,
            ) as span:
                span.set_attributes(
                    {
                        "inspect.message_count_total": len(messages),
                        "inspect.message_count_logged": TRACE_MAX_MESSAGE_SPANS,
                        "inspect.task": task_name,
                        "inspect.eval_id": str(eval_id),
                        "inspect.sample_id": str(sample_id) if sample_id else "",
                    }
                )
                span.set_outputs(
                    {
                        "note": (
                            f"Truncated message spans from {len(messages)} to "
                            f"{TRACE_MAX_MESSAGE_SPANS} to keep trace size manageable."
                        )
                    }
                )

    def _log_model_event_span(
        self: _TracingHost,
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
            span.set_inputs(
                {"messages": TracingMixin._truncate_trace_text(_to_json(input_data))}
            )

            output = _obj_get(event, "output")
            completion = _obj_get(output, "completion")
            usage = _usage_to_dict(_obj_get(output, "usage"))

            span.set_outputs(
                {
                    "completion": TracingMixin._truncate_trace_text(
                        _to_json(completion)
                    ),
                    "usage": usage,
                }
            )

    def _log_tool_event_span(
        self: _TracingHost,
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
                    "arguments": TracingMixin._truncate_trace_text(
                        _to_json(_obj_get(event, "arguments"))
                    ),
                }
            )

            error = _obj_get(event, "error")
            span.set_outputs(
                {
                    "result": TracingMixin._truncate_trace_text(
                        _to_json(_obj_get(event, "result"))
                    ),
                    "error": (
                        TracingMixin._truncate_trace_text(_to_json(error))
                        if error
                        else None
                    ),
                }
            )

    def _log_error_event_span(
        self: _TracingHost,
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
            span.set_outputs(
                {
                    "error": TracingMixin._truncate_trace_text(
                        _to_json(_obj_get(event, "error"))
                    )
                }
            )
