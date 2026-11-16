"""Shared fixtures for the app's tests."""
from __future__ import annotations

import shutil
import sys
from dataclasses import replace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if sys.path[:1] != [str(REPO_ROOT)]:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.datasets import AppPaths, default_paths  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"


# The default paths, with results on a copy of the fixture tree and the real bench/analyst code.
@pytest.fixture
def fixture_paths(tmp_path: Path) -> AppPaths:
    results = tmp_path / "analyst_check"
    shutil.copytree(FIXTURES / "analyst_check", results)
    return replace(default_paths(), results=results, code=REPO_ROOT / "bench/analyst", app_results=tmp_path / "app_results")
