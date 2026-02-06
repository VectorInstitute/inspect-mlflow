"""Shared utility functions for inspect-mlflow."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlparse

from inspect_ai.scorer import CORRECT, INCORRECT, NOANSWER, PARTIAL

# Hardcoded prefix for all MLflow tags, metrics, and params
TAG_PREFIX = "inspect"

# MLflow span types for GenAI tracing
SPAN_TYPE_AGENT = "AGENT"
SPAN_TYPE_LLM = "LLM"
SPAN_TYPE_TOOL = "TOOL"
SPAN_TYPE_CHAIN = "CHAIN"


def _clean_key(value: str) -> str:
    """Clean a string for use as MLflow metric/param key."""
    return (
        value.replace(" ", "_").replace("/", ".").replace(":", ".").replace("\\", ".")
    )


def _clean_token(value: str, max_len: int = 48) -> str:
    """Clean and truncate a string token, adding hash if too long."""
    token = _clean_key(str(value))
    if not token:
        return "empty"
    if len(token) <= max_len:
        return token
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:8]
    return f"{token[: max_len - 9]}_{digest}"


def _coerce_metric(value: Any) -> float | None:
    """Convert a score value to a float metric."""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        val = value.strip()
        if val == CORRECT:
            return 1.0
        if val == PARTIAL:
            return 0.5
        if val == INCORRECT or val == NOANSWER:
            return 0.0
        lower = val.lower()
        if lower in {"yes", "true"}:
            return 1.0
        if lower in {"no", "false"}:
            return 0.0
        try:
            return float(val)
        except ValueError:
            return None
    if hasattr(value, "value"):
        return _coerce_metric(getattr(value, "value"))
    return None


def _to_json(value: Any) -> str | None:
    """Convert value to JSON string for params/artifacts."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    except TypeError:
        return str(value)


def _jsonable(value: Any) -> Any:
    """Convert value to JSON-serializable form."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump())
        except Exception:
            return str(value)
    if hasattr(value, "dict"):
        try:
            return _jsonable(value.dict())
        except Exception:
            return str(value)
    if hasattr(value, "__dict__"):
        try:
            return _jsonable(vars(value))
        except Exception:
            return str(value)
    return str(value)


def _obj_get(obj: Any, key: str, default: Any = None) -> Any:
    """Safely get attribute or dict key."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _iter_scores(scores: Any) -> Iterable[tuple[str, Any]]:
    """Iterate over scores dict or list."""
    if isinstance(scores, dict):
        return scores.items()
    if hasattr(scores, "items"):
        return scores.items()
    if isinstance(scores, (list, tuple)):
        items: list[tuple[str, Any]] = []
        for idx, score in enumerate(scores):
            name = getattr(score, "name", None)
            items.append((name or str(idx), score))
        return items
    return []


def _usage_to_dict(usage: Any) -> dict[str, int]:
    """Extract token usage as dict."""
    if usage is None:
        return {}
    keys = [
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "input_tokens_cache_write",
        "input_tokens_cache_read",
        "reasoning_tokens",
    ]
    output: dict[str, int] = {}
    for key in keys:
        value = _obj_get(usage, key)
        if value is not None:
            try:
                output[key] = int(value)
            except (TypeError, ValueError):
                continue
    return output


def _sum_usage(usage_map: Any) -> dict[str, int]:
    """Sum token usage across multiple models."""
    totals: dict[str, int] = defaultdict(int)
    if not isinstance(usage_map, dict):
        return {}
    for usage in usage_map.values():
        for key, value in _usage_to_dict(usage).items():
            totals[key] += int(value)
    return dict(totals)


def _location_to_local_path(location: str) -> Path | None:
    """Convert log location to local path."""
    parsed = urlparse(location)
    if not parsed.scheme:
        path = Path(location)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    return None
