"""Tests for bench/analyst/run_analyst_check.py. Nothing here runs a real step."""
from __future__ import annotations

import importlib.util
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
ANALYST = HERE.parent
REPO_ROOT = HERE.parents[2]


# Load a module by path under an an_ name, clear of other bench tests.
def _load(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ANALYST / file_name)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


mod = _load("an_run_analyst_check", "run_analyst_check.py")

PY, AI_PY = ".venv/bin/python", ".venv-ai/bin/python"
BUILD = ("cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSHILMANDB_ENABLE_ML=ON "
         "-DSHILMANDB_BUILD_BENCH=ON && cmake --build build --target analyst_bridge")
DEFAULT_COMMANDS = [
    ("prepare", [PY, "bench/analyst/prepare_datasets.py"]),
    ("describe", [PY, "bench/analyst/describe_datasets.py"]),
    ("probe", [PY, "bench/analyst/probe_subset.py"]),
    ("key", [PY, "bench/analyst/answer_key.py", "check"]),
]
GENERATE = ["--steps", "generate", "--split", "dev", "--model", "qwen36-35b"]


# Fail the test if anything tries to start a process.
def _no_process(*args, **kwargs):
    raise AssertionError("a process was started")


# Every test gets a fake bridge that exists and a driver that can't start processes.
@pytest.fixture(autouse=True)
def _guard(tmp_path, monkeypatch):
    bridge = tmp_path / f"analyst_bridge_{os.getpid()}"
    bridge.write_text("")
    monkeypatch.setattr(mod, "BRIDGE", bridge)
    monkeypatch.setattr(mod, "subprocess", SimpleNamespace(Popen=_no_process))


# Swap the driver's Popen for a fake; outcomes maps script names to wait() results.
# Returns the log of ("start" | "wait" | "kill", cmd) events.
def _fake_popen(monkeypatch, outcomes: dict | None = None) -> list[tuple[str, list[str]]]:
    log = []

    # Logs its command and plays back that script's outcomes (default: exit 0).
    class FakePopen:
        # Log the start; the only extra argument allowed is the repo root as cwd.
        def __init__(self, cmd, *args, **kwargs):
            assert not args and kwargs == {"cwd": REPO_ROOT}
            self.cmd = cmd
            self.left = list((outcomes or {}).get(Path(cmd[1]).name, [0]))
            log.append(("start", cmd))

        # Return the next outcome, or raise it when it's an exception class.
        def wait(self):
            log.append(("wait", self.cmd))
            result = self.left.pop(0)
            if isinstance(result, type) and issubclass(result, BaseException):
                raise result
            return result

        # Log the kill.
        def kill(self):
            log.append(("kill", self.cmd))

    monkeypatch.setattr(mod, "subprocess", SimpleNamespace(Popen=FakePopen))
    return log


# Script file names of the commands started, in order.
def _scripts(log) -> list[str]:
    return [Path(cmd[1]).name for kind, cmd in log if kind == "start"]


# Run main, expecting argparse to stop it; returns the exit code.
def _exit_code(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as e:
        mod.main(argv)
    return e.value.code


# The (step, command) pairs a dry run printed, in order.
def _commands(out: str) -> list[tuple[str, list[str]]]:
    rows, step = [], None
    for line in out.splitlines():
        if line.startswith("== dry run"):
            continue
        if line.startswith("== "):
            step = line[3:]
        elif line:
            rows.append((step, shlex.split(line)))
    return rows


# Dry-run main; returns the exit code and the printed commands.
def _dry(argv: list[str], capsys) -> tuple[int, list[tuple[str, list[str]]]]:
    code = mod.main([*argv, "--dry-run"])
    return code, _commands(capsys.readouterr().out)


# Default dry run: the four prep steps on .venv, paths from the repo root.
def test_default_steps_dry_run(capsys):
    code = mod.main(["--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    assert out.splitlines()[0] == f"== dry run, nothing runs; commands run from {REPO_ROOT}"
    assert _commands(out) == DEFAULT_COMMANDS
    for _, cmd in DEFAULT_COMMANDS:
        assert shlex.join(cmd) in out.splitlines()


# generate refuses to start without both --split and --model, dry run or not.
def test_generate_needs_split_and_model(capsys):
    for flags in ([], ["--split", "dev"], ["--model", "qwen36-35b"]):
        for dry in ([], ["--dry-run"]):
            assert mod.main(["--steps", "prepare,generate", *flags, *dry]) == 2
            captured = capsys.readouterr()
            assert captured.out == ""
            assert captured.err.splitlines() == ["error: generate needs --split and --model"]


# generate runs on .venv-ai with split, model and limit; a given python stays as given.
def test_generate_uses_ai_python(capsys):
    assert _dry(GENERATE, capsys) == (0, [("generate", [AI_PY, "bench/analyst/run_pipeline.py", "--split", "dev", "--model", "qwen36-35b"])])

    code, rows = _dry([*GENERATE, "--limit-per-dataset", "4", "--ai-python", "/opt/ai/python", "--python", "py3"], capsys)
    assert code == 0
    assert rows == [("generate", ["/opt/ai/python", "bench/analyst/run_pipeline.py", "--split", "dev",
                                  "--model", "qwen36-35b", "--limit-per-dataset", "4"])]


# Steps run in the fixed order whatever order they're given in, each once.
def test_order_is_fixed(monkeypatch, capsys):
    code, rows = _dry(["--steps", "analyze,prepare,analyze"], capsys)
    assert code == 0
    assert [step for step, _ in rows] == ["prepare", "analyze"]

    log = _fake_popen(monkeypatch)
    assert mod.main(["--steps", "analyze,key,prepare"]) == 0
    assert _scripts(log) == ["prepare_datasets.py", "answer_key.py", "analyze_analyst_check.py"]

    args = SimpleNamespace(steps=["analyze", "prepare"], split=None, model=None, limit_per_dataset=None,
                           python=PY, ai_python=AI_PY, results=None)
    assert [step for step, _ in mod.build_commands(args)] == ["prepare", "analyze"]


# probe, key or generate with no bridge: exit 2 with the build command, nothing started.
def test_missing_bridge_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(mod, "BRIDGE", tmp_path / "missing" / "analyst_bridge")
    log = _fake_popen(monkeypatch)

    for steps in (["--steps", "prepare,probe"], ["--steps", "key"], GENERATE, []):
        for dry in ([], ["--dry-run"]):
            assert mod.main([*steps, *dry]) == 2
            captured = capsys.readouterr()
            assert captured.out == ""
            assert captured.err.splitlines()[-1] == BUILD
            assert "analyst_bridge" in captured.err.splitlines()[0]
    assert log == []

    # Steps that don't use the bridge still run.
    code, rows = _dry(["--steps", "prepare,describe,analyze"], capsys)
    assert code == 0
    assert [step for step, _ in rows] == ["prepare", "describe", "analyze"]


# The first failing step's exit code stops the run; later steps never start.
def test_stops_at_first_failure(monkeypatch, capsys):
    log = _fake_popen(monkeypatch, {"describe_datasets.py": [3]})
    assert mod.main([]) == 3
    assert _scripts(log) == ["prepare_datasets.py", "describe_datasets.py"]

    captured = capsys.readouterr()
    assert re.search(r"^== prepare: done in \d+\.\d s$", captured.out, re.M)
    assert "== describe: start" in captured.out
    assert "== describe: done" not in captured.out
    assert "== probe" not in captured.out
    assert captured.err.splitlines() == ["== describe: failed with exit code 3, stopping"]


# An unknown or empty --steps is an argparse error (exit 2).
def test_unknown_step_exits_2(capsys):
    assert _exit_code(["--steps", "prepare,bogus"]) == 2
    assert "unknown step bogus" in capsys.readouterr().err
    assert _exit_code(["--steps", ","]) == 2
    assert _exit_code(["--steps", "prepare", "--split", "train"]) == 2


# --results goes to every script only when the driver was given it, exactly as given.
def test_results_passed_only_when_given(capsys):
    every = ["--steps", ",".join(mod.STEPS), "--split", "test", "--model", "qwen36-35b"]

    code, rows = _dry(every, capsys)
    assert code == 0
    assert len(rows) == 6
    assert all("--results" not in cmd for _, cmd in rows)

    code, rows = _dry([*every, "--results", "../elsewhere/analyst"], capsys)
    assert code == 0
    assert [step for step, _ in rows] == list(mod.STEPS)
    assert all(cmd[-2:] == ["--results", "../elsewhere/analyst"] and cmd.count("--results") == 1 for _, cmd in rows)


# analyze defaults to test; it gets --model only with --split test.
def test_analyze_gets_model_only_with_test(capsys):
    analyze = [PY, "bench/analyst/analyze_analyst_check.py"]
    assert _dry(["--steps", "analyze"], capsys) == (0, [("analyze", [*analyze, "--split", "test"])])
    assert _dry(["--steps", "analyze", "--model", "m1"], capsys) == (0, [("analyze", [*analyze, "--split", "test", "--model", "m1"])])
    assert _dry(["--steps", "analyze", "--split", "dev", "--model", "m1"], capsys) == (0, [("analyze", [*analyze, "--split", "dev"])])

    code, rows = _dry(["--steps", "generate,analyze", "--split", "dev", "--model", "m1", "--limit-per-dataset", "4"], capsys)
    assert code == 0
    assert rows == [
        ("generate", [AI_PY, "bench/analyst/run_pipeline.py", "--split", "dev", "--model", "m1", "--limit-per-dataset", "4"]),
        ("analyze", [*analyze, "--split", "dev"]),
    ]


# A real run prints start, command and done per step, run from the repo root.
def test_real_run_prints_each_step(monkeypatch, capsys):
    log = _fake_popen(monkeypatch)
    assert mod.main([]) == 0
    assert [cmd for kind, cmd in log if kind == "start"] == [cmd for _, cmd in DEFAULT_COMMANDS]

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 12
    for i, (step, cmd) in enumerate(DEFAULT_COMMANDS):
        assert re.fullmatch(rf"== {step}: start \d\d:\d\d:\d\d", lines[3 * i])
        assert lines[3 * i + 1] == shlex.join(cmd)
        assert re.fullmatch(rf"== {step}: done in \d+\.\d s", lines[3 * i + 2])


# A step killed by a signal stops the run with 128 + the signal number.
def test_signal_exits_128_plus_signal(monkeypatch, capsys):
    log = _fake_popen(monkeypatch, {"probe_subset.py": [-9]})
    assert mod.main([]) == 137
    assert _scripts(log) == ["prepare_datasets.py", "describe_datasets.py", "probe_subset.py"]
    assert capsys.readouterr().err.splitlines() == ["== probe: killed by signal 9, stopping"]


# Ctrl-C waits for the step to exit, says so in one line and exits 130.
def test_ctrl_c_waits_and_exits_130(monkeypatch, capsys):
    log = _fake_popen(monkeypatch, {"describe_datasets.py": [KeyboardInterrupt, 0]})
    assert mod.main([]) == 130
    describe = DEFAULT_COMMANDS[1][1]
    assert log[-3:] == [("start", describe), ("wait", describe), ("wait", describe)]
    assert capsys.readouterr().err.splitlines() == ["== describe: stopped by Ctrl-C"]


# A second Ctrl-C while waiting kills the step.
def test_second_ctrl_c_kills(monkeypatch, capsys):
    log = _fake_popen(monkeypatch, {"prepare_datasets.py": [KeyboardInterrupt, KeyboardInterrupt, -9]})
    assert mod.main([]) == 130
    assert [kind for kind, _ in log] == ["start", "wait", "wait", "kill", "wait"]
    assert capsys.readouterr().err.splitlines() == ["== prepare: stopped by Ctrl-C"]


# A python that can't start stops the run with 1.
def test_cant_start_exits_1(monkeypatch, capsys):
    # Raises the way Popen does for a missing interpreter.
    def missing(cmd, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", cmd[0])

    monkeypatch.setattr(mod, "subprocess", SimpleNamespace(Popen=missing))
    assert mod.main(["--steps", "prepare,describe", "--python", "nope/python"]) == 1
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 1 and err[0].startswith("== prepare: can't start nope/python: ")


# --limit-per-dataset must be a whole number of at least 1.
def test_limit_must_be_positive():
    for bad in ("0", "-1", "x"):
        assert _exit_code([*GENERATE, "--limit-per-dataset", bad]) == 2


# The script itself, run from elsewhere, prints the same repo-relative commands.
def test_script_dry_run_from_another_folder(tmp_path):
    script = ANALYST / "run_analyst_check.py"
    done = subprocess.run([sys.executable, str(script), "--steps", "prepare,describe", "--dry-run"],
                          cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert _commands(done.stdout) == DEFAULT_COMMANDS[:2]
