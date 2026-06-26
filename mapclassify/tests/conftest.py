import pytest


def pytest_configure(config):  # noqa: ARG001
    """PyTest session attributes, methods, etc."""

    pytest.image_comp_kws = {
        "extensions": ["png"],
        "remove_text": True,
        "style": "mpl20",
    }
    pytest.image_comp_kws_legend_text = {
        "extensions": ["png"],
        "remove_text": True,
        "style": "mpl20",
    }
