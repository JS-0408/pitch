# T0.2 — Config and logging unit tests
import os
import pytest
from yaazhi.config import load_config, _Namespace


def test_load_local():
    cfg = load_config("local")
    assert cfg.sensor.width == 160
    assert cfg.sensor.height == 120
    assert cfg.sensor.fps == 9


def test_missing_key_raises():
    cfg = load_config("local")
    with pytest.raises(KeyError, match="nonexistent_key"):
        _ = cfg.nonexistent_key


def test_nested_missing_key_raises():
    cfg = load_config("local")
    with pytest.raises(KeyError):
        _ = cfg.sensor.nonexistent_subkey


def test_config_readonly():
    cfg = load_config("local")
    with pytest.raises(AttributeError):
        cfg.sensor = "bad"


def test_invalid_env_raises():
    with pytest.raises(ValueError, match="Unknown YAAZHI_ENV"):
        load_config("prod")


def test_env_var_override(monkeypatch):
    monkeypatch.setenv("YAAZHI_ENV", "local")
    cfg = load_config()
    assert cfg.sensor.width == 160
