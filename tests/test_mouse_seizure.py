"""Baseline test.

Importing by package name (rather than by file path) is what the src layout
buys us: this resolves the same way from tests, notebooks, and scripts, with
no sys.path manipulation and no dependence on the current directory.
"""

import mouse_seizure


def test_package_imports() -> None:
    assert mouse_seizure.__name__ == "mouse_seizure"


def test_parse_name_namespaces_split() -> None:
    """Training M3 and test M3 are different animals; ids must not collide."""
    from mouse_seizure.io import parse_name

    assert parse_name("Day2_M3") == ("train_M3", "train", "2")
    assert parse_name("20240209_M3_MAT") == ("test_M3", "test", None)


def test_events_handles_censored_boundaries() -> None:
    """A mask starting and ending inside an event still yields paired spans."""
    import numpy as np

    from mouse_seizure.io import events

    mask = np.array([1, 1, 0, 0, 1, 1, 1, 0, 1, 1], dtype=np.uint8)
    ev = events(mask, fs=2.0)
    assert ev.tolist() == [[0.0, 1.0], [2.0, 3.5], [4.0, 5.0]]
    assert (ev[:, 1] > ev[:, 0]).all()
