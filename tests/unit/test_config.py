"""
Unit tests for app/config.py.
Verifies all env vars load correctly with their defaults.
"""
import os
import importlib
from unittest.mock import patch


def _reload_config(env: dict):
    """Reload config module with a custom environment, bypassing .env file."""
    with patch.dict(os.environ, env, clear=True), \
         patch("dotenv.load_dotenv"):
        import app.config as config
        importlib.reload(config)
        return config


def test_openai_model_default():
    config = _reload_config({})
    assert config.OPENAI_MODEL == "gpt-4o"


def test_openai_model_from_env():
    config = _reload_config({"OPENAI_MODEL": "gpt-4-turbo"})
    assert config.OPENAI_MODEL == "gpt-4-turbo"


def test_openai_embedding_model_default():
    config = _reload_config({})
    assert config.OPENAI_EMBEDDING_MODEL == "text-embedding-3-small"


def test_openai_embedding_model_from_env():
    config = _reload_config({"OPENAI_EMBEDDING_MODEL": "text-embedding-ada-002"})
    assert config.OPENAI_EMBEDDING_MODEL == "text-embedding-ada-002"


def test_openai_api_key_default_is_none():
    config = _reload_config({})
    assert config.OPENAI_API_KEY is None


def test_openai_api_key_from_env():
    config = _reload_config({"OPENAI_API_KEY": "sk-test-123"})
    assert config.OPENAI_API_KEY == "sk-test-123"


def test_app_name_default():
    config = _reload_config({})
    assert config.APP_NAME == "Aegis"


def test_app_name_from_env():
    config = _reload_config({"APP_NAME": "AegisTest"})
    assert config.APP_NAME == "AegisTest"


def test_app_env_default():
    config = _reload_config({})
    assert config.APP_ENV == "development"


def test_app_env_from_env():
    config = _reload_config({"APP_ENV": "production"})
    assert config.APP_ENV == "production"


def test_cve_server_path_exists():
    config = _reload_config({})
    assert hasattr(config, "CVE_SERVER_PATH")


def test_cve_server_path_points_to_correct_file():
    config = _reload_config({})
    assert config.CVE_SERVER_PATH.endswith("cve_server.py")


def test_cve_server_path_is_absolute():
    config = _reload_config({})
    assert os.path.isabs(config.CVE_SERVER_PATH)
