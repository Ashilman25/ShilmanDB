"""SF lists of run_sqlite_benchmarks.py and run_shilmandb_benchmarks.py (other runners import them)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

TPCH_DIR = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, TPCH_DIR / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sqlite_runner = load("run_sqlite_benchmarks")
shilmandb_runner = load("run_shilmandb_benchmarks")


def test_sqlite_runner_sf_lists() -> None:
    assert sqlite_runner.ALL_SCALE_FACTORS == ["0.01", "0.1", "1.0"]
    # 100 is known so the reference runners take it; SQLite itself refuses it
    assert sqlite_runner.KNOWN_SCALE_FACTORS == ["0.01", "0.1", "1.0", "10", "100"]


def test_sqlite_runner_rejects_unknown_sf(monkeypatch: pytest.MonkeyPatch,
                                          capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["run_sqlite_benchmarks.py", "--sf", "3"])
    with pytest.raises(SystemExit) as exc:
        sqlite_runner.main()
    assert exc.value.code == 1
    assert "unknown scale factor '3'" in capsys.readouterr().out


def test_sqlite_runner_refuses_sf100(monkeypatch: pytest.MonkeyPatch,
                                     capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["run_sqlite_benchmarks.py", "--sf", "10", "100"])
    with pytest.raises(SystemExit) as exc:
        sqlite_runner.main()
    assert exc.value.code == 1
    assert "SQLite doesn't run at SF=100" in capsys.readouterr().out


def test_shilmandb_runner_sf_lists() -> None:
    assert shilmandb_runner.ALL_SCALE_FACTORS == ["0.01", "0.1", "1.0"]
    assert shilmandb_runner.KNOWN_SCALE_FACTORS == ["0.01", "0.1", "1.0", "10"]
    assert shilmandb_runner.POOL_SIZES["10"] == 8192
