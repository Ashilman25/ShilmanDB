"""Tests for bench/analyst/freeze.py."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ANALYST = HERE.parent


# Load a module by path under an an_ name, clear of other bench tests.
def _load(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ANALYST / file_name)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


mod = _load("an_freeze", "freeze.py")
an = mod.an

CONFIGS = {
    "gemma4-26b": {"path": "mlx-community/gemma-4-26b-a4b-it-4bit", "thinking": False, "max_tokens": 1024},
    "qwen36-35b": {"path": "mlx-community/Qwen3.6-35B-A3B-4bit", "thinking": False, "max_tokens": 1024},
}
RESULTS_NAMES = [
    "questions.jsonl", "datasets/tpch/description.md", "datasets/reviews/description.md", "probe.json", "subset_rules.md",
]
CODE_NAMES = [
    "prompts.py", "prompt_examples.json", "sql_facts.py", "checks.py", "scoring.py",
    "datasets/tpch.json", "datasets/reviews.json", "an_common.py", "describe_datasets.py", "model_mlx.py",
]
ALL_NAMES = RESULTS_NAMES + CODE_NAMES


# A results folder and a code folder holding every frozen file.
def _folders(tmp_path: Path) -> tuple[Path, Path]:
    res, code = tmp_path / f"res_{os.getpid()}", tmp_path / f"code_{os.getpid()}"
    for base, names in ((res, RESULTS_NAMES), (code, CODE_NAMES)):
        for name in names:
            path = base / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"original {name}\n", encoding="utf-8")
    return res, code


def _read_freeze(res: Path) -> dict:
    return json.loads((res / "freeze.json").read_text(encoding="utf-8"))


def _write_freeze(res: Path, record) -> None:
    (res / "freeze.json").write_text(json.dumps(record), encoding="utf-8")


def _base(name: str, res: Path, code: Path) -> Path:
    return res if name in RESULTS_NAMES else code


# A test run's answers file under the results folder.
def _test_run(res: Path, config: str = "qwen36-35b") -> Path:
    path = an.paths(res).run_file("test", config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")
    return path


def test_frozen_lists():
    assert mod.FROZEN_RESULTS == RESULTS_NAMES
    assert mod.FROZEN_CODE == CODE_NAMES
    assert set(mod.FROZEN_RESULTS).isdisjoint(mod.FROZEN_CODE)  # names key one flat map


def test_compute_shape(tmp_path):
    res, code = _folders(tmp_path)
    record = mod.compute(res, code, "qwen36-35b", CONFIGS)
    assert set(record) == {"files", "model"}
    assert set(record["files"]) == set(ALL_NAMES)
    for name in ALL_NAMES:
        assert record["files"][name] == an.sha1_file(_base(name, res, code) / name)
    assert record["model"] == {"id": "qwen36-35b", **CONFIGS["qwen36-35b"]}
    assert not (res / "freeze.json").exists()  # compute writes nothing


def test_compute_names_every_missing_file(tmp_path):
    res, code = _folders(tmp_path)
    (code / "checks.py").unlink()
    (code / "sql_facts.py").unlink()
    (res / "datasets/tpch/description.md").unlink()
    with pytest.raises(FileNotFoundError) as e:
        mod.compute(res, code, "qwen36-35b", CONFIGS)
    for name in ("checks.py", "sql_facts.py", "description.md"):
        assert name in str(e.value)
    assert "prompts.py" not in str(e.value)


def test_compute_unknown_config(tmp_path):
    res, code = _folders(tmp_path)
    with pytest.raises(ValueError, match="'nope'.*gemma4-26b"):
        mod.compute(res, code, "nope", CONFIGS)


def test_write_then_verify_clean(tmp_path):
    res, code = _folders(tmp_path)
    path = mod.write(res, code, "qwen36-35b", CONFIGS)
    assert path == res / "freeze.json"
    frozen = _read_freeze(res)
    assert {k: v for k, v in frozen.items() if k != "written_at"} == mod.compute(res, code, "qwen36-35b", CONFIGS)
    assert datetime.fromisoformat(frozen["written_at"]).tzinfo is not None
    assert mod.verify(res, code, CONFIGS) == []
    assert mod.verify(str(res), str(code), CONFIGS) == []  # plain strings work too


def test_changed_file_is_reported(tmp_path):
    res, code = _folders(tmp_path)
    mod.write(res, code, "qwen36-35b", CONFIGS); (code / "checks.py").write_text("changed")
    assert mod.verify(res, code, CONFIGS) == ["checks.py changed"]


@pytest.mark.parametrize("name", ALL_NAMES)
def test_every_frozen_file_is_watched(tmp_path, name):
    res, code = _folders(tmp_path)
    mod.write(res, code, "qwen36-35b", CONFIGS)
    (_base(name, res, code) / name).write_text("edited\n", encoding="utf-8")
    assert mod.verify(res, code, CONFIGS) == [f"{name} changed"]


def test_unfrozen_file_is_ignored(tmp_path):
    res, code = _folders(tmp_path)
    mod.write(res, code, "qwen36-35b", CONFIGS)
    (code / "run_pipeline.py").write_text("new\n", encoding="utf-8")
    (res / "key_answers.json").write_text("{}\n", encoding="utf-8")
    assert mod.verify(res, code, CONFIGS) == []


def test_code_after_the_literal_is_caught(tmp_path):
    res, code = _folders(tmp_path)
    model_file = code / "model_mlx.py"
    model_file.write_text(f"MODEL_CONFIGS = {CONFIGS!r}\n", encoding="utf-8")
    mod.write(res, code, "qwen36-35b", mod.read_model_configs(model_file))

    with model_file.open("a", encoding="utf-8") as f:
        f.write("MODEL_CONFIGS['qwen36-35b']['max_tokens'] = 2048\n")
    configs = mod.read_model_configs(model_file)
    assert configs["qwen36-35b"]["max_tokens"] == 1024  # ast only sees the literal
    assert mod.verify(res, code, configs) == ["model_mlx.py changed"]


def test_write_refuses_while_a_test_run_exists(tmp_path):
    res, code = _folders(tmp_path)
    runs = [_test_run(res, "qwen36-35b"), _test_run(res, "gemma4-26b")]
    runs[1].write_text("", encoding="utf-8")  # even an empty run counts
    with pytest.raises(FileExistsError) as e:
        mod.write(res, code, "qwen36-35b", CONFIGS)
    assert all(str(run) in str(e.value) for run in runs)
    assert not (res / "freeze.json").exists()


def test_refreeze_over_a_test_run_keeps_the_old_freeze(tmp_path):
    res, code = _folders(tmp_path)
    mod.write(res, code, "qwen36-35b", CONFIGS)
    before = (res / "freeze.json").read_bytes()
    _test_run(res)

    (code / "checks.py").write_text("changed", encoding="utf-8")
    with pytest.raises(FileExistsError, match="test_qwen36-35b.jsonl"):
        mod.write(res, code, "qwen36-35b", CONFIGS)
    assert (res / "freeze.json").read_bytes() == before
    assert mod.verify(res, code, CONFIGS) == ["checks.py changed"]


def test_dev_runs_do_not_block_a_freeze(tmp_path):
    res, code = _folders(tmp_path)
    dev = an.paths(res).run_file("dev", "qwen36-35b")
    dev.parent.mkdir(parents=True)
    dev.write_text("{}\n", encoding="utf-8")
    mod.write(res, code, "qwen36-35b", CONFIGS)
    assert mod.verify(res, code, CONFIGS) == []


def test_frozen_file_now_missing(tmp_path):
    res, code = _folders(tmp_path)
    mod.write(res, code, "qwen36-35b", CONFIGS)
    (code / "checks.py").unlink()
    (res / "probe.json").write_text("edited\n", encoding="utf-8")
    assert mod.verify(res, code, CONFIGS) == ["probe.json changed", "checks.py is missing"]


def test_changed_model_config_is_reported(tmp_path):
    res, code = _folders(tmp_path)
    mod.write(res, code, "qwen36-35b", CONFIGS)

    changed = copy.deepcopy(CONFIGS)
    changed["qwen36-35b"]["max_tokens"] = 2048
    assert mod.verify(res, code, changed) == ["model qwen36-35b changed: max_tokens 1024 -> 2048"]

    changed["qwen36-35b"]["thinking"] = True
    del changed["qwen36-35b"]["path"]
    changed["qwen36-35b"]["top_p"] = 0.9
    assert mod.verify(res, code, changed) == [
        'model qwen36-35b changed: max_tokens 1024 -> 2048, path "mlx-community/Qwen3.6-35B-A3B-4bit" -> unset, '
        "thinking false -> true, top_p unset -> 0.9"
    ]


def test_model_config_compared_strictly(tmp_path):
    res, code = _folders(tmp_path)
    configs = {"m": {"path": "x", "thinking": False, "opts": {"a": 1, "b": 2}}}
    mod.write(res, code, "m", configs)
    assert mod.verify(res, code, {"m": {"path": "x", "thinking": False, "opts": {"b": 2, "a": 1}}}) == []
    assert mod.verify(res, code, {"m": {"path": "x", "thinking": 0, "opts": {"a": 1, "b": 2}}}) == [
        "model m changed: thinking false -> 0"
    ]


def test_other_model_config_change_is_ignored(tmp_path):
    res, code = _folders(tmp_path)
    mod.write(res, code, "qwen36-35b", CONFIGS)
    changed = copy.deepcopy(CONFIGS)
    changed["gemma4-26b"]["max_tokens"] = 2048
    changed["new-model"] = {"path": "x", "thinking": False, "max_tokens": 1024}
    assert mod.verify(res, code, changed) == []


def test_frozen_model_no_longer_in_configs(tmp_path):
    res, code = _folders(tmp_path)
    mod.write(res, code, "qwen36-35b", CONFIGS)
    gone = {k: v for k, v in CONFIGS.items() if k != "qwen36-35b"}
    assert mod.verify(res, code, gone) == ["model qwen36-35b is no longer in MODEL_CONFIGS"]


def test_config_cannot_override_its_id(tmp_path):
    res, code = _folders(tmp_path)
    configs = {"q": {"id": "other", "path": "x"}}
    assert mod.compute(res, code, "q", configs)["model"] == {"id": "q", "path": "x"}
    mod.write(res, code, "q", configs)
    assert _read_freeze(res)["model"]["id"] == "q"
    assert mod.verify(res, code, configs) == []
    assert mod.frozen_model_id(res) == "q"


def test_tuple_in_config_survives_json(tmp_path):
    res, code = _folders(tmp_path)
    configs = {"m": {"path": "x", "stop": ("a", "b")}}
    mod.write(res, code, "m", configs)
    assert mod.verify(res, code, configs) == []


def test_missing_freeze(tmp_path):
    res, code = _folders(tmp_path)
    assert mod.verify(res, code, CONFIGS) == ["missing freeze.json"]


def test_freeze_json_that_does_not_parse(tmp_path):
    res, code = _folders(tmp_path)
    (res / "freeze.json").write_text('{"files": ', encoding="utf-8")
    problems = mod.verify(res, code, CONFIGS)
    assert len(problems) == 1 and problems[0].startswith("freeze.json doesn't parse: ")

    (res / "freeze.json").write_bytes(b"\xff\xfe{}")
    problems = mod.verify(res, code, CONFIGS)
    assert len(problems) == 1 and problems[0].startswith("freeze.json doesn't parse: ")

    _write_freeze(res, ["not", "an", "object"])
    assert mod.verify(res, code, CONFIGS) == ["freeze.json isn't a JSON object"]


def test_unreadable_freeze_json(tmp_path):
    res, code = _folders(tmp_path)
    mod.write(res, code, "qwen36-35b", CONFIGS)
    path = res / "freeze.json"
    path.chmod(0)
    try:
        if os.access(path, os.R_OK):
            pytest.skip("this user can read a file with no permissions")
        problems = mod.verify(res, code, CONFIGS)
        assert len(problems) == 1 and problems[0].startswith("can't read freeze.json: ")
        assert mod.frozen_model_id(res) is None
    finally:
        path.chmod(0o644)


def test_frozen_model_id(tmp_path):
    res, code = _folders(tmp_path)
    assert mod.frozen_model_id(res) is None
    mod.write(res, code, "qwen36-35b", CONFIGS)
    assert mod.frozen_model_id(res) == "qwen36-35b"
    assert mod.frozen_model_id(str(res)) == "qwen36-35b"

    good = _read_freeze(res)
    for broken in (["a list"], {"files": good["files"]}, {**good, "model": "qwen36-35b"}, {**good, "model": {"id": 5}}):
        _write_freeze(res, broken)
        assert mod.frozen_model_id(res) is None
    (res / "freeze.json").write_text('{"model": ', encoding="utf-8")
    assert mod.frozen_model_id(res) is None


def test_freeze_json_missing_fields(tmp_path):
    res, code = _folders(tmp_path)
    mod.write(res, code, "qwen36-35b", CONFIGS)
    good = _read_freeze(res)

    _write_freeze(res, {k: v for k, v in good.items() if k != "files"})
    assert mod.verify(res, code, CONFIGS) == ["freeze.json has no files"]

    _write_freeze(res, {**good, "files": ["checks.py"]})
    assert mod.verify(res, code, CONFIGS) == ["freeze.json has no files"]

    _write_freeze(res, {k: v for k, v in good.items() if k != "model"})
    assert mod.verify(res, code, CONFIGS) == ["freeze.json has no model id"]

    _write_freeze(res, {**good, "model": {k: v for k, v in good["model"].items() if k != "id"}})
    assert mod.verify(res, code, CONFIGS) == ["freeze.json has no model id"]

    _write_freeze(res, {"written_at": good["written_at"]})
    assert mod.verify(res, code, CONFIGS) == ["freeze.json has no files", "freeze.json has no model id"]


def test_names_freeze_json_and_lists_disagree_on(tmp_path):
    res, code = _folders(tmp_path)
    mod.write(res, code, "qwen36-35b", CONFIGS)
    good = _read_freeze(res)

    files = {k: v for k, v in good["files"].items() if k != "prompts.py"}
    _write_freeze(res, {**good, "files": files})
    assert mod.verify(res, code, CONFIGS) == ["prompts.py isn't in freeze.json"]

    _write_freeze(res, {**good, "files": {**good["files"], "old.py": "0" * 40}})
    assert mod.verify(res, code, CONFIGS) == ["old.py is in freeze.json but no longer frozen"]


def test_real_configs_read_without_mlx():
    configs = mod.read_model_configs(ANALYST / "model_mlx.py")
    assert "gemma4-26b" in configs
    assert configs == _load("an_model_mlx", "model_mlx.py").MODEL_CONFIGS


def test_configs_read_without_running_the_file(tmp_path):
    path = tmp_path / f"model_{os.getpid()}.py"
    path.write_text(f"import mlx_that_is_not_installed\n\nMODEL_CONFIGS = {CONFIGS!r}\n", encoding="utf-8")
    assert mod.read_model_configs(path) == CONFIGS

    path.write_text("MODEL_CONFIGS: dict = {'a': {'path': 'x'}}\nMODEL_CONFIGS = {'b': {'path': 'y'}}\n", encoding="utf-8")
    assert mod.read_model_configs(path) == {"b": {"path": "y"}}  # the last one, as Python would


@pytest.mark.parametrize("source", [
    "OTHER = {'a': {}}\n",
    "def f():\n    MODEL_CONFIGS = {'a': {}}\n",
    "MODEL_CONFIGS = dict(a={})\n",
    "MODEL_CONFIGS = ['a']\n",
    "MODEL_CONFIGS = {'a': 1}\n",
])
def test_bad_model_configs(tmp_path, source):
    path = tmp_path / f"model_{os.getpid()}.py"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError, match="MODEL_CONFIGS"):
        mod.read_model_configs(path)


# The command line, on a code folder that also holds a model_mlx.py.
def _cli_folders(tmp_path: Path) -> tuple[Path, Path]:
    res, code = _folders(tmp_path)
    (code / "model_mlx.py").write_text(f"MODEL_CONFIGS = {CONFIGS!r}\n", encoding="utf-8")
    return res, code


def test_cli_write_and_verify(tmp_path, capsys):
    res, code = _cli_folders(tmp_path)
    assert mod.main(["verify", "--results", str(res)], code_dir=code) == 1
    assert "missing freeze.json" in capsys.readouterr().err

    assert mod.main(["write", "--model", "qwen36-35b", "--results", str(res)], code_dir=code) == 0
    assert str(res / "freeze.json") in capsys.readouterr().out
    assert _read_freeze(res)["model"]["id"] == "qwen36-35b"

    assert mod.main(["verify", "--results", str(res)], code_dir=code) == 0
    capsys.readouterr()

    (code / "checks.py").write_text("changed", encoding="utf-8")
    (code / "model_mlx.py").write_text(f"MODEL_CONFIGS = {{'gemma4-26b': {CONFIGS['gemma4-26b']!r}}}\n", encoding="utf-8")
    assert mod.main(["verify", "--results", str(res)], code_dir=code) == 1
    err = capsys.readouterr().err
    assert "checks.py changed" in err and "model qwen36-35b is no longer in MODEL_CONFIGS" in err


def test_cli_refusals(tmp_path, capsys):
    res, code = _cli_folders(tmp_path)
    assert mod.main(["write", "--model", "nope", "--results", str(res)], code_dir=code) == 1
    err = capsys.readouterr().err
    assert "'nope'" in err and "Traceback" not in err
    assert not (res / "freeze.json").exists()

    (code / "checks.py").unlink()
    assert mod.main(["write", "--model", "qwen36-35b", "--results", str(res)], code_dir=code) == 1
    err = capsys.readouterr().err
    assert "checks.py" in err and "Traceback" not in err
    assert not (res / "freeze.json").exists()

    (code / "model_mlx.py").unlink()
    assert mod.main(["verify", "--results", str(res)], code_dir=code) == 1
    err = capsys.readouterr().err
    assert "model_mlx.py" in err and "Traceback" not in err


def test_cli_refuses_over_a_test_run(tmp_path, capsys):
    res, code = _cli_folders(tmp_path)
    run = _test_run(res)
    assert mod.main(["write", "--model", "qwen36-35b", "--results", str(res)], code_dir=code) == 1
    err = capsys.readouterr().err
    assert str(run) in err and "Traceback" not in err
    assert not (res / "freeze.json").exists()

    run.unlink()
    assert mod.main(["write", "--model", "qwen36-35b", "--results", str(res)], code_dir=code) == 0
    before = (res / "freeze.json").read_bytes()
    _test_run(res, "gemma4-26b")
    assert mod.main(["write", "--model", "gemma4-26b", "--results", str(res)], code_dir=code) == 1
    assert "test_gemma4-26b.jsonl" in capsys.readouterr().err
    assert (res / "freeze.json").read_bytes() == before


def test_cli_needs_a_model(tmp_path):
    res, code = _cli_folders(tmp_path)
    with pytest.raises(SystemExit) as e:
        mod.main(["write", "--results", str(res)], code_dir=code)
    assert e.value.code == 2
