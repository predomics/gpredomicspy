"""Basic tests for gpredomicspy."""

import pytest


def test_import():
    """Test that the package can be imported."""
    import gpredomicspy
    assert hasattr(gpredomicspy, "Param")
    assert hasattr(gpredomicspy, "fit")
    assert hasattr(gpredomicspy, "__version__")


def test_param_creation():
    """Test creating a Param with defaults."""
    from gpredomicspy import Param
    p = Param()
    assert repr(p).startswith("Param(")


def test_param_set():
    """Test setting parameters."""
    from gpredomicspy import Param
    p = Param()
    p.set("max_epochs", 10)
    p.set("population_size", 500)
    p.set_string("algo", "ga")
    p.set_bool("gpu", False)


def test_param_set_invalid():
    """Test that invalid parameter names raise errors."""
    from gpredomicspy import Param
    p = Param()
    with pytest.raises(ValueError):
        p.set("nonexistent_param", 42)
