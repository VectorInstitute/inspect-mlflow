"""MLflow configuration settings."""

from __future__ import annotations

import json
from typing import Annotated, Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def _empty_to_none(v: Any) -> Any:
    """Convert empty strings to None."""
    if isinstance(v, str) and v.strip() == "":
        return None
    return v


class MLflowSettings(BaseSettings):
    """MLflow integration settings.

    Configuration is loaded from environment variables with INSPECT_MLFLOW_ prefix.
    Settings can also be overridden via task metadata with inspect_mlflow_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="INSPECT_MLFLOW_",
        env_file=".env" if __name__ != "__main__" else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Override to allow passing _env_file=None for testing
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @field_validator("*", mode="before")
    @classmethod
    def empty_string_to_none(cls, v: Any, info: Any) -> Any:
        """Convert empty strings to None for optional fields."""
        return _empty_to_none(v)

    # Core settings
    enabled: bool = Field(
        default=True,
        description="Enable/disable MLflow logging entirely",
    )
    tracking_uri: str | None = Field(
        default=None,
        description="MLflow tracking server URI (e.g., http://localhost:5000 or sqlite:///mlflow.db)",
    )
    experiment: str | None = Field(
        default=None,
        description="MLflow experiment name. Auto-generated from task name if not set.",
    )
    run_name: str | None = Field(
        default=None,
        description="MLflow run name. Uses Inspect run_id if not set.",
    )

    # Logging granularity
    log_artifacts: bool = Field(
        default=True,
        description="Log artifacts (Inspect logs, score tables, etc.)",
    )
    log_traces: bool = Field(
        default=True,
        description="Enable MLflow tracing for samples and events",
    )

    # Autolog settings for automatic LLM call tracing
    autolog_enabled: bool = Field(
        default=True,
        description="Enable automatic tracing of LLM library calls (OpenAI, Anthropic, etc.)",
    )
    autolog_models: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["openai", "anthropic", "langchain", "litellm"],
        description="LLM libraries to autolog. Options: openai, anthropic, langchain, litellm, mistral, etc.",
    )

    @field_validator("autolog_models", mode="before")
    @classmethod
    def parse_autolog_models(cls, v: Any) -> list[str]:
        """Parse autolog_models from comma-separated string, JSON array, or list."""
        if isinstance(v, str):
            value = v.strip()
            if not value:
                return ["openai", "anthropic", "langchain", "litellm"]

            # Support both JSON-array env format and CSV env format.
            if value.startswith("["):
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, list):
                    return [str(m).strip().lower() for m in parsed if str(m).strip()]

            return [m.strip().lower() for m in value.split(",") if m.strip()]
        if isinstance(v, list):
            return [str(m).strip().lower() for m in v]
        return ["openai", "anthropic", "langchain", "litellm"]

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any] | None = None) -> "MLflowSettings":
        """Create settings with optional metadata overrides.

        Task metadata keys with 'inspect_mlflow_' prefix will override env vars.
        Example: metadata={"inspect_mlflow_enabled": False} disables logging.
        """
        overrides = {}
        if metadata:
            prefix = "inspect_mlflow_"
            overrides = {
                k[len(prefix) :]: v
                for k, v in metadata.items()
                if k.lower().startswith(prefix)
            }
        return cls(**overrides)
