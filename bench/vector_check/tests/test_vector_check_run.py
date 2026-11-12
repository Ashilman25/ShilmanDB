"""Tests for bench/vector_check/run_vector_check.py. Nothing here runs a real step."""
from __future__ import annotations

import importlib.util
import os
import shlex
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]


# Load a module by path under a vc_ name, clear of other bench tests.
def _load(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, HERE.parent / file_name)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


mod = _load("vc_run", "run_vector_check.py")

ALL_STEPS = ["prepare", "queries", "embed", "truth", "load", "grid", "graph",
             "rem", "trajectories", "train", "replay", "analyze"]
STOPPING = ("rem", "trajectories", "train", "replay")


# tmp_path with links resolved, as the driver prints it.
@pytest.fixture
def tmp(tmp_path):
    return tmp_path.resolve()


# Fail the test if anything tries to start a process.
def _no_subprocess(*args, **kwargs):
    raise AssertionError("a process was started")


# Swap subprocess.Popen for a fake; outcomes(cmd) lists what each wait() returns or raises.
# Returns the log of ("start" | "wait" | "kill", cmd) events.
def _fake_popen(monkeypatch, outcomes) -> list[tuple[str, list[str]]]:
    log = []

    # Logs its command and plays back that command's outcomes.
    class FakePopen:
        # Log the start; any extra argument would mean a shell or a capture.
        def __init__(self, cmd, *args, **kwargs):
            assert not args and not kwargs
            self.cmd, self.left = cmd, list(outcomes(cmd))
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

    monkeypatch.setattr(mod.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(mod.subprocess, "run", _no_subprocess)
    return log


# The commands started, in order, from a fake Popen log.
def _started(log) -> list[list[str]]:
    return [cmd for kind, cmd in log if kind == "start"]


# Script file names of the commands started.
def _scripts(log) -> list[str]:
    return [Path(cmd[1]).name for cmd in _started(log)]


# Results and ai_queries folders under tmp, as driver flags.
def _folders(tmp: Path) -> list[str]:
    return ["--results", str(tmp / "res"), "--ai", str(tmp / "ai")]


# Run main, expecting argparse to stop it; returns the exit code.
def _exit_code(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as e:
        mod.main(argv)
    return e.value.code


# Dry-run main and return its exit code and printed output.
def _dry(argv: list[str], monkeypatch, capsys) -> tuple[int, str]:
    monkeypatch.setattr(mod.subprocess, "run", _no_subprocess)
    monkeypatch.setattr(mod.subprocess, "Popen", _no_subprocess)
    code = mod.main([*argv, "--dry-run"])
    return code, capsys.readouterr().out


# The printed lines grouped by step: (step, command list or None, note or None).
def _parse(out: str) -> list[tuple[str, list[str] | None, str | None]]:
    rows, step = [], None
    for line in out.splitlines():
        if line.startswith("== ") and ": start " in line:
            step = line[3:line.index(":")]
        elif line.startswith("== "):
            continue
        elif line.startswith("note: "):
            rows.append((step, None, line))
        elif line:
            rows.append((step, shlex.split(line), None))
    return rows


# Commands only, as (step, command) pairs.
def _commands(out: str) -> list[tuple[str, list[str]]]:
    return [(s, c) for s, c, _ in _parse(out) if c is not None]


# The value after a flag in a command.
def _flag(cmd: list[str], name: str) -> str:
    return cmd[cmd.index(name) + 1]


# A file under tmp standing in for a python interpreter.
def _fake_python(tmp: Path, name: str) -> Path:
    path = tmp / f"{name}_{os.getpid()}"
    path.write_text("")
    return path


# 70K dry run: twelve steps, embed on .venv-ai, no slice.
def test_dry_run_70k_commands(tmp, monkeypatch, capsys):
    code, out = _dry(["--size", "70k", *_folders(tmp)], monkeypatch, capsys)
    assert code == 0
    starts = [line[3:line.index(":")] for line in out.splitlines() if ": start " in line]
    dones = [line for line in out.splitlines() if line.startswith("== ") and ": done in " in line]
    assert starts == ALL_STEPS and len(dones) == 12

    cmds = _commands(out)
    assert [s for s, _ in cmds] == ALL_STEPS
    by_step = dict(cmds)
    python = str(REPO_ROOT / ".venv" / "bin" / "python")
    ai_python = str(REPO_ROOT / ".venv-ai" / "bin" / "python")
    for step, cmd in cmds:
        assert cmd[0] == (ai_python if step == "embed" else python)
        assert Path(cmd[1]).is_absolute() and Path(cmd[1]).exists()
    assert "slice" not in out

    assert by_step["prepare"][1:] == [str(HERE.parent / "prepare_vector_data.py"), "table", "--size", "70k",
                                      "--results", str(tmp / "res"), "--ai", str(tmp / "ai")]
    assert by_step["queries"][2] == "queries"
    assert by_step["embed"][1].endswith("embed_vector_check.py") and by_step["embed"][2] == "queries"
    assert by_step["truth"][1].endswith("vector_truth.py")
    assert by_step["load"][1].endswith("pg_vector_load.py")
    assert by_step["grid"][1].endswith("pg_vector_grid.py")
    assert by_step["graph"][1].endswith("hnsw_graph.py")
    for step in STOPPING:
        assert by_step[step][1].endswith("stopping_check.py") and by_step[step][2] == step
    assert by_step["analyze"][1:] == [str(HERE.parent / "analyze_vector_check.py"), "--results", str(tmp / "res")]
    for step in ALL_STEPS[:-1]:
        if step not in ("queries", "embed"):
            assert _flag(by_step[step], "--size") == "70k"


# 500K dry run embeds the queries, then the slice without --ai.
def test_dry_run_500k_embeds_slice(tmp, monkeypatch, capsys):
    code, out = _dry(["--size", "500k", *_folders(tmp)], monkeypatch, capsys)
    assert code == 0
    embeds = [c for s, c in _commands(out) if s == "embed"]
    assert [c[2] for c in embeds] == ["queries", "slice"]
    assert all(c[0] == str(REPO_ROOT / ".venv-ai" / "bin" / "python") for c in embeds)
    assert embeds[1][3:] == ["--results", str(tmp / "res")]
    assert all(_flag(c, "--size") == "500k" for s, c in _commands(out) if s not in ("queries", "embed", "analyze"))


# --queries outside 1-100 at 70K or 30-50 at 500K exits 2.
def test_queries_bounds(tmp, monkeypatch, capsys):
    base = _folders(tmp)
    for size, n in (("500k", "29"), ("500k", "51")):
        assert _exit_code(["--size", size, "--queries", n, *base, "--dry-run"]) == 2
        assert "--queries at 500k must be 30-50 (it may only go down)" in capsys.readouterr().err
    for n in ("0", "101"):
        assert _exit_code(["--size", "70k", "--queries", n, *base, "--dry-run"]) == 2
        assert "--queries at 70k must be 1-100" in capsys.readouterr().err
    for size, n in (("500k", "40"), ("500k", "30"), ("500k", "50"), ("70k", "1"), ("70k", "100")):
        code, out = _dry(["--size", size, "--queries", n, *base], monkeypatch, capsys)
        assert code == 0
        assert _flag(dict(_commands(out))["truth"], "--queries") == n


# An unknown or empty step list exits 2 and names the step.
def test_unknown_step_rejected(tmp, capsys):
    assert _exit_code(["--size", "70k", "--steps", "prepare,bogus", *_folders(tmp), "--dry-run"]) == 2
    assert "bogus" in capsys.readouterr().err
    assert _exit_code(["--size", "70k", "--steps", ",", *_folders(tmp), "--dry-run"]) == 2


# A failing truth step stops the run with its exit code and a resume line.
def test_stops_at_first_failure(tmp, monkeypatch, capsys):
    log = _fake_popen(monkeypatch, lambda cmd: [3 if cmd[1].endswith("vector_truth.py") else 0])
    py, ai_py = _fake_python(tmp, "py"), _fake_python(tmp, "aipy")
    code = mod.main(["--size", "70k", *_folders(tmp), "--python", str(py), "--ai-python", str(ai_py)])
    assert code == 3
    assert _scripts(log) == ["prepare_vector_data.py", "prepare_vector_data.py", "embed_vector_check.py",
                             "vector_truth.py"]
    captured = capsys.readouterr()
    assert "truth" in captured.err and "exit code 3" in captured.err
    assert "to resume: rerun with --steps truth,load,grid,graph,rem,trajectories,train,replay,analyze " \
           "(and your other flags)" in captured.err.splitlines()
    assert "== truth: done" not in captured.out and "== load: start" not in captured.out


# The resume line lists the failed step and the later selected steps only.
def test_resume_hint_lists_later_selected_steps(tmp, monkeypatch, capsys):
    log = _fake_popen(monkeypatch, lambda cmd: [5 if cmd[1].endswith("pg_vector_grid.py") else 0])
    py, ai_py = _fake_python(tmp, "py"), _fake_python(tmp, "aipy")
    argv = ["--size", "70k", "--steps", "analyze,grid,load,train", *_folders(tmp),
            "--python", str(py), "--ai-python", str(ai_py)]
    assert mod.main(argv) == 5
    assert _scripts(log) == ["pg_vector_load.py", "pg_vector_grid.py"]
    err = capsys.readouterr().err.splitlines()
    assert "to resume: rerun with --steps grid,train,analyze (and your other flags)" in err

    # No resume line when every command passes.
    log = _fake_popen(monkeypatch, lambda cmd: [0])
    assert mod.main(argv) == 0
    assert "to resume" not in capsys.readouterr().err


# Finished queries and embeddings print notes instead of commands.
def test_skip_notes_for_existing_outputs(tmp, monkeypatch, capsys):
    res = tmp / "res"
    (res / "data").mkdir(parents=True)
    (res / "embeddings").mkdir()
    csv_path, npy_path = res / "data" / "query_reviews.csv", res / "embeddings" / "query_reviews.npy"
    csv_path.write_text("x\n")
    npy_path.write_text("x")
    (res / "embeddings" / "slice.npy").write_text("x")
    os.utime(csv_path, (1_000_000, 1_000_000))
    os.utime(npy_path, (2_000_000, 2_000_000))

    code, out = _dry(["--size", "500k", *_folders(tmp)], monkeypatch, capsys)
    assert code == 0
    rows = _parse(out)
    assert ("queries", None, "note: data/query_reviews.csv exists; pass --redo to resample") in rows
    assert ("embed", None, "note: embeddings/query_reviews.npy is newer than data/query_reviews.csv; "
                           "pass --redo to re-embed") in rows
    assert ("embed", None, "note: embeddings/slice.npy exists; pass --redo to re-embed") in rows
    assert [s for s, _ in _commands(out)] == [s for s in ALL_STEPS if s not in ("queries", "embed")]

    # An embedding older than its reviews is redone.
    os.utime(npy_path, (500_000, 500_000))
    code, out = _dry(["--size", "70k", *_folders(tmp)], monkeypatch, capsys)
    assert [c[2] for s, c in _commands(out) if s == "embed"] == ["queries"]
    assert "note: data/query_reviews.csv exists" in out


# With no reviews file, both the queries and the embedding run.
def test_embed_runs_when_reviews_file_is_missing(tmp, monkeypatch, capsys):
    (tmp / "res" / "embeddings").mkdir(parents=True)
    (tmp / "res" / "embeddings" / "query_reviews.npy").write_text("x")
    code, out = _dry(["--size", "70k", "--steps", "queries,embed", *_folders(tmp)], monkeypatch, capsys)
    assert code == 0 and "note:" not in out
    assert [(s, c[2]) for s, c in _commands(out)] == [("queries", "queries"), ("embed", "queries")]


# --redo reaches only the grid and runs every skipped command.
def test_redo_goes_to_grid_and_turns_off_skips(tmp, monkeypatch, capsys):
    res = tmp / "res"
    (res / "data").mkdir(parents=True)
    (res / "embeddings").mkdir()
    for path in (res / "data" / "query_reviews.csv", res / "embeddings" / "query_reviews.npy",
                 res / "embeddings" / "slice.npy"):
        path.write_text("x")
    os.utime(res / "data" / "query_reviews.csv", (1_000_000, 1_000_000))

    code, out = _dry(["--size", "500k", "--redo", *_folders(tmp)], monkeypatch, capsys)
    assert code == 0 and "note:" not in out
    cmds = _commands(out)
    assert [s for s, _ in cmds] == ALL_STEPS[:3] + ["embed"] + ALL_STEPS[3:]
    assert [s for s, c in cmds if "--redo" in c] == ["grid"]

    code, out = _dry(["--size", "500k", *_folders(tmp)], monkeypatch, capsys)
    assert all("--redo" not in c for _, c in _commands(out))


# --ai goes everywhere but the slice embed and the analyzer.
def test_ai_only_where_the_script_takes_it(tmp, monkeypatch, capsys):
    code, out = _dry(["--size", "500k", *_folders(tmp)], monkeypatch, capsys)
    for step, cmd in _commands(out):
        if step == "analyze" or (step == "embed" and cmd[2] == "slice"):
            assert "--ai" not in cmd
        else:
            assert _flag(cmd, "--ai") == str(tmp / "ai")
        assert _flag(cmd, "--results") == str(tmp / "res")


# --jobs reaches rem, trajectories and replay only, with the core-based default.
def test_jobs_only_on_rem_trajectories_replay(tmp, monkeypatch, capsys):
    code, out = _dry(["--size", "70k", "--jobs", "5", *_folders(tmp)], monkeypatch, capsys)
    assert {s: _flag(c, "--jobs") for s, c in _commands(out) if "--jobs" in c} == \
        {"rem": "5", "trajectories": "5", "replay": "5"}

    code, out = _dry(["--size", "70k", "--steps", "rem", *_folders(tmp)], monkeypatch, capsys)
    assert _flag(_commands(out)[0][1], "--jobs") == str(max(1, (os.cpu_count() or 1) - 2))


# --part2-limit reaches all four stopping commands as --limit.
def test_limit_on_all_four_stopping_steps(tmp, monkeypatch, capsys):
    code, out = _dry(["--size", "70k", "--part2-limit", "200,100,100", *_folders(tmp)], monkeypatch, capsys)
    assert {s: _flag(c, "--limit") for s, c in _commands(out) if "--limit" in c} == \
        {s: "200,100,100" for s in STOPPING}

    code, out = _dry(["--size", "70k", *_folders(tmp)], monkeypatch, capsys)
    assert all("--limit" not in c for _, c in _commands(out))


# --queries reaches truth and the grid only.
def test_queries_only_on_truth_and_grid(tmp, monkeypatch, capsys):
    code, out = _dry(["--size", "70k", "--queries", "4", *_folders(tmp)], monkeypatch, capsys)
    assert {s: _flag(c, "--queries") for s, c in _commands(out) if "--queries" in c} == {"truth": "4", "grid": "4"}

    code, out = _dry(["--size", "70k", *_folders(tmp)], monkeypatch, capsys)
    assert all("--queries" not in c for _, c in _commands(out))


# Steps run in their fixed order, each once, however they're given.
def test_steps_run_in_fixed_order_once(tmp, monkeypatch, capsys):
    code, out = _dry(["--size", "70k", "--steps", "analyze,truth,truth,prepare", *_folders(tmp)],
                     monkeypatch, capsys)
    assert code == 0
    assert [s for s, _ in _commands(out)] == ["prepare", "truth", "analyze"]
    assert out.count("== truth: start") == 1


# A bad --part2-limit or --jobs exits 2.
def test_bad_part2_limit_and_jobs(tmp, capsys):
    base = ["--size", "70k", *_folders(tmp), "--dry-run"]
    for bad in ("200,100", "1,2,3,4", "0,1,1", "a,b,c", "200,,100"):
        assert _exit_code([*base, "--part2-limit", bad]) == 2
        assert "--part2-limit" in capsys.readouterr().err
    for bad in ("0", "-1", "x"):
        assert _exit_code([*base, "--jobs", bad]) == 2
        assert "--jobs" in capsys.readouterr().err


# --size is needed unless the only step is analyze.
def test_size_needed_unless_only_analyze(tmp, monkeypatch, capsys):
    assert _exit_code([*_folders(tmp), "--dry-run"]) == 2
    assert "--size" in capsys.readouterr().err
    assert _exit_code(["--steps", "truth,analyze", *_folders(tmp), "--dry-run"]) == 2

    code, out = _dry(["--steps", "analyze", *_folders(tmp)], monkeypatch, capsys)
    assert code == 0
    assert [s for s, _ in _commands(out)] == ["analyze"]
    assert "--size" not in _commands(out)[0][1]


# From another folder every printed path is absolute and still right.
def test_runs_from_another_directory(tmp, monkeypatch, capsys):
    monkeypatch.chdir(tmp)
    code, out = _dry(["--size", "500k", "--results", "res", "--ai", "ai"], monkeypatch, capsys)
    assert code == 0
    for _, cmd in _commands(out):
        assert Path(cmd[0]).is_absolute() and Path(cmd[1]).is_absolute() and Path(cmd[1]).exists()
        assert _flag(cmd, "--results") == str(tmp.resolve() / "res")
        if "--ai" in cmd:
            assert _flag(cmd, "--ai") == str(tmp.resolve() / "ai")


# The default results and ai_queries folders print as absolute paths.
def test_default_folders_are_absolute(tmp, monkeypatch, capsys):
    monkeypatch.chdir(tmp)
    args = mod.parse_args(["--size", "70k"])
    commands = mod.build_commands(args)
    prepare = next(cmd for step, cmd, _ in commands if step == "prepare")
    assert _flag(prepare, "--results") == str(mod.vc.RESULTS.resolve())
    assert _flag(prepare, "--ai") == str(mod.vc.AI_ROOT.resolve())


# A relative --python becomes absolute but keeps its link, as a venv needs.
def test_relative_python_is_made_absolute_without_following_links(tmp, monkeypatch, capsys):
    real = _fake_python(tmp, "real")
    (tmp / "venv_link").symlink_to(real)
    monkeypatch.chdir(tmp)
    code, out = _dry(["--size", "70k", "--steps", "prepare", "--python", "venv_link", *_folders(tmp)],
                     monkeypatch, capsys)
    assert _commands(out)[0][1][0] == str(tmp / "venv_link")


# When every command passes, each starts once, in the printed order, and is waited for.
def test_every_command_runs_in_order_when_all_pass(tmp, monkeypatch, capsys):
    log = _fake_popen(monkeypatch, lambda cmd: [0])
    py, ai_py = _fake_python(tmp, "py"), _fake_python(tmp, "aipy")
    argv = ["--size", "500k", *_folders(tmp), "--python", str(py), "--ai-python", str(ai_py)]
    assert mod.main(argv) == 0
    out = capsys.readouterr().out
    assert _started(log) == [c for _, c in _commands(out)]
    assert [kind for kind, _ in log] == ["start", "wait"] * 13
    assert out.count(": done in ") == 12


# A missing interpreter stops the run before any command starts.
def test_missing_python_stops_before_anything_runs(tmp, monkeypatch, capsys):
    monkeypatch.setattr(mod.subprocess, "run", _no_subprocess)
    monkeypatch.setattr(mod.subprocess, "Popen", _no_subprocess)
    py = _fake_python(tmp, "py")
    code = mod.main(["--size", "70k", *_folders(tmp), "--python", str(py),
                     "--ai-python", str(tmp / f"missing_{os.getpid()}")])
    assert code == 1
    assert "missing_" in capsys.readouterr().err


# Ctrl-C in the grid waits for its cleanup, exits 130, runs nothing after.
def test_ctrl_c_names_the_step(tmp, monkeypatch, capsys):
    log = _fake_popen(monkeypatch, lambda cmd: [KeyboardInterrupt, 1] if cmd[1].endswith("pg_vector_grid.py") else [0])
    py, ai_py = _fake_python(tmp, "py"), _fake_python(tmp, "aipy")
    code = mod.main(["--size", "70k", *_folders(tmp), "--python", str(py), "--ai-python", str(ai_py)])
    assert code == 130
    assert _scripts(log) == ["prepare_vector_data.py", "prepare_vector_data.py", "embed_vector_check.py",
                             "vector_truth.py", "pg_vector_load.py", "pg_vector_grid.py"]
    assert [kind for kind, _ in log[-3:]] == ["start", "wait", "wait"]
    captured = capsys.readouterr()
    assert "== grid: stopped by Ctrl-C" in captured.err
    assert "to resume: rerun with --steps grid,graph,rem,trajectories,train,replay,analyze " \
           "(and your other flags)" in captured.err.splitlines()
    assert "== grid: done" not in captured.out and "== graph: start" not in captured.out


# A second Ctrl-C while the command cleans up kills it, and still exits 130.
def test_second_ctrl_c_kills_the_command(tmp, monkeypatch, capsys):
    log = _fake_popen(monkeypatch, lambda cmd: [KeyboardInterrupt, KeyboardInterrupt, -9])
    py, ai_py = _fake_python(tmp, "py"), _fake_python(tmp, "aipy")
    code = mod.main(["--size", "70k", "--steps", "load,grid", *_folders(tmp),
                     "--python", str(py), "--ai-python", str(ai_py)])
    assert code == 130
    assert [kind for kind, _ in log] == ["start", "wait", "wait", "kill", "wait"]
    assert _scripts(log) == ["pg_vector_load.py"]
    assert "to resume: rerun with --steps load,grid (and your other flags)" in capsys.readouterr().err


# A command that can't start exits 1, names the step and says how to resume.
def test_unstartable_command_exits_1(tmp, monkeypatch, capsys):
    # Act as if the interpreter can't be executed.
    def no_start(cmd, *args, **kwargs):
        raise PermissionError(13, "Permission denied", cmd[0])

    monkeypatch.setattr(mod.subprocess, "Popen", no_start)
    monkeypatch.setattr(mod.subprocess, "run", _no_subprocess)
    py, ai_py = _fake_python(tmp, "py"), _fake_python(tmp, "aipy")
    code = mod.main(["--size", "70k", "--steps", "truth,load", *_folders(tmp),
                     "--python", str(py), "--ai-python", str(ai_py)])
    assert code == 1
    err = capsys.readouterr().err
    assert "== truth: can't start" in err
    assert "to resume: rerun with --steps truth,load (and your other flags)" in err.splitlines()


# Ctrl-C while a command is starting still gives the stop and resume lines.
def test_ctrl_c_while_a_command_starts(tmp, monkeypatch, capsys):
    started = []

    # Act as if Ctrl-C arrived while the grid's process was starting.
    def popen(cmd, *args, **kwargs):
        started.append(Path(cmd[1]).name)
        raise KeyboardInterrupt

    monkeypatch.setattr(mod.subprocess, "Popen", popen)
    monkeypatch.setattr(mod.subprocess, "run", _no_subprocess)
    py, ai_py = _fake_python(tmp, "py"), _fake_python(tmp, "aipy")
    code = mod.main(["--size", "70k", "--steps", "grid,graph", *_folders(tmp),
                     "--python", str(py), "--ai-python", str(ai_py)])
    assert code == 130
    assert started == ["pg_vector_grid.py"]
    captured = capsys.readouterr()
    assert "== grid: stopped by Ctrl-C" in captured.err
    assert "to resume: rerun with --steps grid,graph (and your other flags)" in captured.err.splitlines()
    assert "== graph: start" not in captured.out


# Run every step at 70K, failing the named script; returns the resume line.
def _resume_line(tmp, monkeypatch, capsys, failing: str, redo: bool) -> str:
    # prepare's table command always passes, so a prepare failure is its queries command.
    _fake_popen(monkeypatch, lambda cmd: [7 if Path(cmd[1]).name == failing and cmd[2:3] != ["table"] else 0])
    py, ai_py = _fake_python(tmp, "py"), _fake_python(tmp, "aipy")
    argv = ["--size", "70k", *_folders(tmp), "--python", str(py), "--ai-python", str(ai_py)]
    assert mod.main([*argv, "--redo"] if redo else argv) == 7
    return next(line for line in capsys.readouterr().err.splitlines() if line.startswith("to resume"))


# With --redo, resuming after embed warns that --redo would delete the grid's cells.
def test_resume_hint_drops_redo_after_embed(tmp, monkeypatch, capsys):
    drop = " (and your other flags except --redo, which would delete the grid's finished cells)"
    later = "grid,graph,rem,trajectories,train,replay,analyze"
    assert _resume_line(tmp, monkeypatch, capsys, "pg_vector_grid.py", True) == \
        f"to resume: rerun with --steps {later}{drop}"
    assert _resume_line(tmp, monkeypatch, capsys, "vector_truth.py", True) == \
        f"to resume: rerun with --steps truth,load,{later}{drop}"
    assert _resume_line(tmp, monkeypatch, capsys, "analyze_vector_check.py", True) == \
        f"to resume: rerun with --steps analyze{drop}"


# Up to embed, or without --redo, the resume line keeps the plain ending.
def test_resume_hint_keeps_plain_ending(tmp, monkeypatch, capsys):
    plain = " (and your other flags)"
    assert _resume_line(tmp, monkeypatch, capsys, "embed_vector_check.py", True) == \
        "to resume: rerun with --steps embed,truth,load,grid,graph,rem,trajectories,train,replay,analyze" + plain
    assert _resume_line(tmp, monkeypatch, capsys, "prepare_vector_data.py", True).endswith(
        "--steps queries,embed,truth,load,grid,graph,rem,trajectories,train,replay,analyze" + plain)
    assert _resume_line(tmp, monkeypatch, capsys, "pg_vector_grid.py", False) == \
        "to resume: rerun with --steps grid,graph,rem,trajectories,train,replay,analyze" + plain


# A command killed by a signal gives 128 plus the signal number.
def test_killed_command_gives_shell_style_code(tmp, monkeypatch, capsys):
    _fake_popen(monkeypatch, lambda cmd: [-9])
    py, ai_py = _fake_python(tmp, "py"), _fake_python(tmp, "aipy")
    code = mod.main(["--size", "70k", "--steps", "prepare", *_folders(tmp),
                     "--python", str(py), "--ai-python", str(ai_py)])
    assert code == 137
    assert "signal 9" in capsys.readouterr().err
