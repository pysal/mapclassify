import pytest


def pytest_configure(config):  # noqa: ARG001
    """PyTest session attributes, methods, etc."""

    pytest.image_comp_kws = {"extensions": ["png"]}
