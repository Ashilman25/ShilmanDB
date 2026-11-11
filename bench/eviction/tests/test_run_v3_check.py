"""Unit tests for bench/eviction/v3_check/run_v3_check.py. Nothing here runs a binary."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SPEC_PATH = Path(__file__).resolve().parents[1] / "v3_check" / "run_v3_check.py"
spec = importlib.util.spec_from_file_location("run_v3_check", SPEC_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules["run_v3_check"] = mod
spec.loader.exec_module(mod)


def make_paths() -> "mod.Paths":
    return mod.Paths(
        load_tpch=Path("build/bench/load_tpch"), sim=Path("build/bench/eviction_sim"),
        samples=Path("build/bench/eviction_samples"), python=Path("py"), traces=Path("T"),
        sim_dir=Path("S5"), out=Path("O"), data_dir=Path("bench/tpch_data"),
    )


def flag(cmd, name):
    return cmd[cmd.index(name) + 1]


def pools(sf):
    return mod.pools(sf, mod.SF_DEFAULTS[sf]["fractions"])


def test_pools_match_recorded_pools():
    assert pools("0.01") == [145, 362, 724]
    assert pools("0.1") == [1442, 3604, 7208]
    assert pools("1.0") == [36008]


def test_owners_cmd_uses_fresh_db_and_indexes():
    cmds = mod.owners_cmds("0.1", make_paths())
    assert cmds[0] == ["rm", "-f", "/tmp/v3_owners_sf0.1.db"]
    assert "--create-indexes" in cmds[1] and flag(cmds[1], "--create-indexes") == "true"
    assert flag(cmds[1], "--dump-page-owners") == "O/sf0.1/page_owners.csv"


def write_owners(path: Path, n: int, kind_last: str = "leaf") -> None:
    lines = ["page_id,kind,object,owner"] + [f"{i},table,t,0" for i in range(n - 1)]
    lines.append(f"{n - 1},{kind_last},x,1")
    path.write_text("\n".join(lines) + "\n")


def test_check_owner_dump_rejects_mismatch(tmp_path):
    f = tmp_path / "owners.csv"
    write_owners(f, 18286)
    mod.check_owner_dump(f, 18286)
    with pytest.raises(ValueError):
        mod.check_owner_dump(f, 18285 + 2)
    write_owners(f, 18285)
    with pytest.raises(ValueError):
        mod.check_owner_dump(f, 18286)
    write_owners(f, 18286, kind_last="other")
    with pytest.raises(ValueError):
        mod.check_owner_dump(f, 18286)


def test_samples_cmds_only_training_traces():
    cmds = mod.samples_cmds("0.1", make_paths(), pools("0.1"), 16384)
    assert len(cmds) == 8 * 3
    for c in cmds:
        assert not any("mixed" in a or "_s3" in a for a in c)
        assert flag(c, "--replay") in ("lru", "belady")


def test_replay_cmds_one_pool_one_model_each():
    cells = list(mod.CELLS)
    cmds = mod.replay_cmds("0.1", make_paths(), pools("0.1"), mod.TEST_TRACES, cells)
    assert len(cmds) == 120
    for c in cmds:
        pool = flag(c, "--pools")
        assert "," not in pool
        cell = flag(c, "--learned-name")
        assert flag(c, "--model") == f"O/sf0.1/models/{cell}_p{pool}.txt"
        assert flag(c, "--jobs") == "1" and "--page-owners" in c


def test_time_cmds_on_mixed_s1_at_10pct():
    cmds = mod.time_cmds("0.1", make_paths(), pools("0.1"), list(mod.CELLS))
    assert len(cmds) == 8
    for c in cmds:
        assert "index_mixed_s1" in c and flag(c, "--pools") == "1442" and "--time-decisions" in c


def test_sf1_requires_cells(capsys):
    with pytest.raises(SystemExit) as e:
        mod.main(["--sf", "1.0", "--dry-run"])
    assert e.value.code != 0
    assert "--cells" in capsys.readouterr().err
    assert mod.main(["--sf", "1.0", "--dry-run", "--cells", "gbdt_richer", "--steps", "replay"]) == 0
    out = capsys.readouterr().out
    assert "--pools 36008" in out and "index_mixed_s3" in out
    assert "index_stream_s3" not in out and "index_lookup_s3" not in out


def test_dry_run_prints_every_step(capsys, tmp_path):
    out = tmp_path / "o"
    steps = "owners,samples,train,validate,replay,time,analyze"
    assert mod.main(["--sf", "0.1", "--dry-run", "--steps", steps, "--out-dir", str(out)]) == 0
    text = capsys.readouterr().out
    for needle in ("--dump-page-owners", "eviction_samples", "train_v3_check.py", "sampled_belady",
                   "--learned-name", "--time-decisions", "analyze_v3_check.py"):
        assert needle in text
    assert not out.exists()


def test_ablation_runs_after_time(capsys, tmp_path):
    steps = list(mod.ALL_STEPS)
    assert steps.index("time") < steps.index("ablation") < steps.index("analyze")
    assert "ablation" not in mod.SF_DEFAULTS["1.0"]["steps"]


def test_ablation_dry_run_uses_placeholder(capsys, tmp_path):
    assert mod.main(["--sf", "0.01", "--dry-run", "--steps", "ablation", "--out-dir", str(tmp_path / "o")]) == 0
    assert "<best richer cell>_no_owner" in capsys.readouterr().out
