"""Unit tests for bench/charts/make_charts.py."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SPEC_PATH = Path(__file__).resolve().parents[1] / "make_charts.py"
spec = importlib.util.spec_from_file_location("make_charts", SPEC_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules["make_charts"] = mod
spec.loader.exec_module(mod)

CONFIGS = [("baseline", "shilmandb"), ("+index", "shilmandb"), ("+vectorized", "shilmandb"),
           ("sqlite_disk", "sqlite"), ("sqlite_memory", "sqlite"), ("duckdb", "duckdb"),
           ("duckdb_1thread", "duckdb"), ("postgres", "postgres"),
           ("postgres_1worker", "postgres")]
QUERIES = ["Q1", "Q3", "Q5", "Q6", "Q10", "Q12", "Q14", "Q19"]
NAMES = ["latency", "vectorized", "eviction", "ceilings", "analyst"]


def write_baseline(path: Path, scale: float) -> None:
    lines = ["query,engine,config,median_ms,status,max_abs_diff,max_rel_diff"]
    for qi, q in enumerate(QUERIES):
        for ci, (config, engine) in enumerate(CONFIGS):
            status = "SKIP" if engine == "sqlite" else "PASS"
            ms = f"{scale * (1 + qi) * (1 + ci):.3f}"
            diffs = ","
            if q == "Q1" and config == "duckdb":
                status, diffs = "DRIFT", "1e-09,2e-12"
            if q == "Q3" and config == "postgres":
                status, ms = "FAIL", ""
            lines.append(f"{q},{engine},{config},{ms},{status},{diffs}")
    path.write_text("\n".join(lines) + "\n")


# A full docs/data folder with every chart's input.
def make_data(root: Path, with_sf10: bool = True) -> Path:
    d = root / "data"
    d.mkdir()
    write_baseline(d / "baseline_sf1.csv", 10.0)
    if with_sf10:
        write_baseline(d / "baseline_sf10.csv", 100.0)
    ev = ["sf,kind,pool_fraction,policy,misses,vs_belady"]
    for kind in ("index_stream", "index_mixed"):
        for frac in ("0.0142", "0.05", "0.1", "0.25", "0.5", "1.0"):
            for pol, m in (("lru", 2.0e6), ("arc_visit", 1.9e6), ("lirs_visit", 1.5e6),
                           ("belady", 1.0e6)):
                ev.append(f"1.0,{kind},{frac},{pol},{m * (1.1 - float(frac))},1.2")
        ev.append(f"1.0,{kind},0.0142,v2,2.1e6,1.1")
    (d / "eviction_misses.csv").write_text("\n".join(ev) + "\n")
    ce = ["part,sf,item,method,ratio"]
    for q, r in (("Q3", 1.01), ("Q5", 1.0), ("Q10", 1.02)):
        ce += [f"join,1.0,{q},classical,{r}", f"join,1.0,{q},learned,{r}"]
    for item, b, v in (("stream 10%", 1.13, 1.2), ("mixed 25%", 1.48, 1.62)):
        ce += [f"eviction,0.01,{item},best_classical,{b}", f"eviction,0.01,{item},v2,{v}"]
    (d / "ceilings.csv").write_text("\n".join(ce) + "\n")
    (d / "analyst_bars.csv").write_text(
        "bar,count,total,rate,wilson_low,wilson_high,needs\n"
        '"catch (bar 1): wrong shown answers warned or refused",20,35,0.571,0.409,0.720,at least 0.50\n'
        '"keep, refusals (bar 2): right shown answers refused",0,227,0.000,0.000,0.017,at most 0.10\n'
        '"keep, warnings (bar 3): right shown answers warned or refused",18,227,0.079,0.051,0.122,at most 0.25\n'
    )
    return d


def test_make_charts_writes_light_and_dark(tmp_path: Path) -> None:
    data = make_data(tmp_path)
    out = tmp_path / "images"
    mod.main(["--data", str(data), "--out", str(out)])
    for name in NAMES:
        for theme in ("light", "dark"):
            png = out / f"{name}-{theme}.png"
            assert png.exists(), png
            assert png.stat().st_size > 0


def test_latency_chart_without_sf10(tmp_path: Path) -> None:
    data = make_data(tmp_path, with_sf10=False)
    out = tmp_path / "images"
    written = mod.chart_latency(data, out)
    assert sorted(p.name for p in written) == ["latency-dark.png", "latency-light.png"]
    assert all(p.stat().st_size > 0 for p in written)


def test_charts_skip_when_baseline_missing(tmp_path: Path) -> None:
    data = make_data(tmp_path)
    (data / "baseline_sf1.csv").unlink()
    (data / "baseline_sf10.csv").unlink()
    out = tmp_path / "images"
    mod.main(["--data", str(data), "--out", str(out)])
    names = sorted(p.name for p in out.iterdir())
    assert names == sorted(f"{n}-{t}.png" for n in NAMES[2:] for t in ("light", "dark"))


def test_fail_row_with_a_time_is_never_drawn(tmp_path: Path) -> None:
    path = tmp_path / "baseline_sf1.csv"
    path.write_text(
        "query,engine,config,median_ms,status,max_abs_diff,max_rel_diff\n"
        "Q1,shilmandb,+index,10.0,PASS,,\n"
        "Q1,shilmandb,+vectorized,6.0,FAIL,,\n"
        "Q3,shilmandb,+index,20.0,PASS,,\n"
        "Q3,shilmandb,+vectorized,10.0,DRIFT,1e-9,1e-12\n"
        "Q5,shilmandb,+index,,FAIL,,\n"
    )
    rows = mod.read_csv(path)
    by_cell = {(r["query"], r["config"]): r for r in rows}
    assert mod.drawable_ms(by_cell[("Q1", "+vectorized")]) is None   # FAIL with a time
    assert mod.drawable_ms(by_cell[("Q5", "+index")]) is None        # FAIL, empty time
    assert mod.drawable_ms(None) is None
    assert mod.drawable_ms(by_cell[("Q3", "+vectorized")]) == 10.0   # DRIFT is drawn
    assert mod.vectorized_ratios(rows) == [("Q3", 0.5)]


def test_needs_text_accepts_only_at_least_and_at_most() -> None:
    assert mod.needs_text("at least 0.50") == ("≥", 0.5)
    assert mod.needs_text("at most 0.10") == ("≤", 0.1)
    with pytest.raises(ValueError):
        mod.needs_text("exactly 0.10")
    with pytest.raises(ValueError):
        mod.needs_text("below 0.25")
