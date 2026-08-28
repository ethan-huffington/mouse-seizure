"""Baseline test.

Importing by package name (rather than by file path) is what the src layout
buys us: this resolves the same way from tests, notebooks, and scripts, with
no sys.path manipulation and no dependence on the current directory.
"""

import mouse_seizure


def test_package_imports() -> None:
    assert mouse_seizure.__name__ == "mouse_seizure"
