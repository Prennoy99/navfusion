"""Smoke test: the package installs and every subpackage imports without optional extras."""

import importlib

import pytest

import navfusion


def test_version() -> None:
    assert navfusion.__version__ == "0.1.0"


@pytest.mark.parametrize("sub", ["data", "filters", "models", "evaluation", "visualization"])
def test_subpackages_import(sub: str) -> None:
    importlib.import_module(f"navfusion.{sub}")
