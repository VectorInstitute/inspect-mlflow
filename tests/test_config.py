"""Tests for MLflowSettings configuration."""

from inspect_mlflow.config import MLflowSettings


class TestMLflowSettings:
    """Test MLflowSettings configuration loading."""

    def test_default_values(self, clean_env):
        """Test that default values are set correctly."""
        # Use _env_file=None to avoid reading .env file in tests
        settings = MLflowSettings(_env_file=None)

        assert settings.enabled is True
        assert settings.tracking_uri is None
        assert settings.experiment is None
        assert settings.run_name is None
        assert settings.log_artifacts is True
        assert settings.log_traces is True
        assert settings.autolog_enabled is True
        assert "openai" in settings.autolog_models
        assert "anthropic" in settings.autolog_models

    def test_env_var_override(self, clean_env, monkeypatch):
        """Test that environment variables override defaults."""
        monkeypatch.setenv("INSPECT_MLFLOW_ENABLED", "false")
        monkeypatch.setenv("INSPECT_MLFLOW_TRACKING_URI", "http://localhost:5000")
        monkeypatch.setenv("INSPECT_MLFLOW_EXPERIMENT", "my-experiment")
        monkeypatch.setenv("INSPECT_MLFLOW_AUTOLOG_ENABLED", "false")

        settings = MLflowSettings(_env_file=None)

        assert settings.enabled is False
        assert settings.tracking_uri == "http://localhost:5000"
        assert settings.experiment == "my-experiment"
        assert settings.autolog_enabled is False

    def test_autolog_models_from_json_array(self, clean_env, monkeypatch):
        """Test parsing autolog_models from JSON array string."""
        # Pydantic-settings expects JSON for list types
        monkeypatch.setenv(
            "INSPECT_MLFLOW_AUTOLOG_MODELS", '["openai","anthropic","mistral"]'
        )

        settings = MLflowSettings(_env_file=None)

        assert settings.autolog_models == ["openai", "anthropic", "mistral"]

    def test_autolog_models_from_csv_env(self, clean_env, monkeypatch):
        """Test parsing autolog_models from comma-separated env var."""
        monkeypatch.setenv("INSPECT_MLFLOW_AUTOLOG_MODELS", "openai,anthropic,mistral")

        settings = MLflowSettings(_env_file=None)

        assert settings.autolog_models == ["openai", "anthropic", "mistral"]

    def test_autolog_models_direct_override(self, clean_env):
        """Test autolog_models with direct override."""
        settings = MLflowSettings(_env_file=None, autolog_models=["openai", "mistral"])

        assert settings.autolog_models == ["openai", "mistral"]

    def test_from_metadata_empty(self, clean_env):
        """Test from_metadata with no overrides."""
        settings = MLflowSettings.from_metadata(None)
        assert settings.enabled is True

        settings = MLflowSettings.from_metadata({})
        assert settings.enabled is True

    def test_from_metadata_with_overrides(self, clean_env):
        """Test from_metadata with task metadata overrides."""
        metadata = {
            "inspect_mlflow_enabled": False,
            "inspect_mlflow_experiment": "override-exp",
            "other_key": "ignored",
        }

        settings = MLflowSettings.from_metadata(metadata)

        assert settings.enabled is False
        assert settings.experiment == "override-exp"

    def test_from_metadata_case_insensitive_prefix(self, clean_env):
        """Test that metadata prefix matching is case-insensitive."""
        metadata = {
            "INSPECT_MLFLOW_enabled": False,
            "Inspect_Mlflow_Experiment": "test",
        }

        settings = MLflowSettings.from_metadata(metadata)

        # Note: The current implementation uses .lower() so it should work
        assert settings.enabled is False

    def test_boolean_env_vars(self, clean_env, monkeypatch):
        """Test various boolean representations in env vars."""
        # Test 'true' values
        for val in ["1", "true", "yes", "True", "TRUE", "on"]:
            monkeypatch.setenv("INSPECT_MLFLOW_ENABLED", val)
            settings = MLflowSettings(_env_file=None)
            assert settings.enabled is True, f"Failed for value: {val}"

        # Test 'false' values
        for val in ["0", "false", "no", "False", "FALSE", "off"]:
            monkeypatch.setenv("INSPECT_MLFLOW_ENABLED", val)
            settings = MLflowSettings(_env_file=None)
            assert settings.enabled is False, f"Failed for value: {val}"

    def test_empty_string_treated_as_none(self, clean_env, monkeypatch):
        """Test that empty strings are treated as None for optional fields."""
        monkeypatch.setenv("INSPECT_MLFLOW_TRACKING_URI", "")
        monkeypatch.setenv("INSPECT_MLFLOW_EXPERIMENT", "")
        monkeypatch.setenv("INSPECT_MLFLOW_RUN_NAME", "")

        settings = MLflowSettings(_env_file=None)

        assert settings.tracking_uri is None
        assert settings.experiment is None
        assert settings.run_name is None
