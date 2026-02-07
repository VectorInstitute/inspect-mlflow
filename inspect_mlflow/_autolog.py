"""Autolog utilities for MLflow hook integration."""

from __future__ import annotations

import importlib
import importlib.util
import logging
from collections.abc import Callable
from typing import Any

_LOG = logging.getLogger(__name__)

# provider -> (mlflow flavor module, function name)
AUTOLOG_MAP: dict[str, tuple[str, str]] = {
    "openai": ("mlflow.openai", "autolog"),
    "anthropic": ("mlflow.anthropic", "autolog"),
    "langchain": ("mlflow.langchain", "autolog"),
    "litellm": ("mlflow.litellm", "autolog"),
    "mistral": ("mlflow.mistral", "autolog"),
    "groq": ("mlflow.groq", "autolog"),
    "cohere": ("mlflow.cohere", "autolog"),
    "gemini": ("mlflow.gemini", "autolog"),
    "bedrock": ("mlflow.bedrock", "autolog"),
}

# providers where dependency module name differs from provider key
DEPENDENCY_MAP: dict[str, str] = {
    "gemini": "google.generativeai",
    "bedrock": "boto3",
}


def enable_autolog(
    models: list[str],
    *,
    find_spec: Callable[[str], Any] = importlib.util.find_spec,
    import_module: Callable[[str], Any] = importlib.import_module,
) -> bool:
    """Enable MLflow autolog for selected model providers.

    Returns True if at least one provider was enabled.
    """
    enabled_any = False

    for model in models:
        model_lower = model.lower()
        if model_lower not in AUTOLOG_MAP:
            continue

        module_name, func_name = AUTOLOG_MAP[model_lower]

        # Require MLflow flavor support and provider SDK.
        if find_spec(module_name) is None:
            continue

        lib_name = DEPENDENCY_MAP.get(model_lower, model_lower)
        if find_spec(lib_name) is None:
            continue

        try:
            module = import_module(module_name)
            autolog_func = getattr(module, func_name, None)
            if autolog_func is not None:
                autolog_func(log_traces=True)
                enabled_any = True
                _LOG.debug(f"Enabled MLflow autolog for {model}")
        except Exception:
            _LOG.debug(f"Could not enable autolog for {model}", exc_info=True)

    return enabled_any
