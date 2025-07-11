import pytest


def pytest_configure(config):  # noqa: ARG001
    """PyTest session attributes, methods, etc."""

    pytest.image_comp_kws = {"extensions": ["png"], "tol": 0.05, "remove_text": True}
    pytest.image_comp_kws_legend_text = {
        "extensions": ["png"],
        "tol": 2.8,
        "remove_text": True,
    }
