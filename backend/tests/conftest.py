"""Shared fixtures.

Environment variables must be set BEFORE the app modules are imported
(settings are read at import time), so the storage/database used by tests
is isolated from the dev database.  conftest is imported first, so setting
os.environ at module top-level is sufficient.
"""

from __future__ import annotations

import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="aqua-test-")
os.environ["AQUA_STORAGE_DIR"] = _TMP
os.environ["AQUA_DB_PATH"] = os.path.join(_TMP, "test.db")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))


@pytest.fixture(scope="session")
def tmp_storage() -> str:
    return _TMP


@pytest.fixture(scope="session")
def synthetic_pipe(tmp_path_factory) -> np.ndarray:
    """A small synthetic sonar scene with a pipe + acoustic shadow."""
    sys.path.insert(0, str(BACKEND / "scripts"))
    import synth

    img, _gt = synth.render("pipe", 11, shape=(240, 256))
    return img


@pytest.fixture(scope="session")
def synthetic_net(tmp_path_factory) -> np.ndarray:
    sys.path.insert(0, str(BACKEND / "scripts"))
    import synth

    img, _gt = synth.render("ghost_net", 21, shape=(240, 256))
    return img


@pytest.fixture(scope="session")
def sample_png(tmp_path_factory, synthetic_pipe) -> Path:
    p = tmp_path_factory.mktemp("media") / "pipe.png"
    import cv2

    cv2.imwrite(str(p), synthetic_pipe)
    return p
