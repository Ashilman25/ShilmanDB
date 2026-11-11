"""Tests for analyze_switch_check.py on synthetic CSVs."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SPEC_PATH = Path(__file__).resolve().parents[1] / "switch_check" / "analyze_switch_check.py"
spec = importlib.util.spec_from_file_location("analyze_switch_check", SPEC_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules["analyze_switch_check"] = mod
spec.loader.exec_module(mod)

POOLS = {0.10: 145, 0.25: 362, 0.50: 724}
TRACES = ("phases_s1", "phases_s2", "phases_s3")
RES_HEADER = ["trace", "switcher", "policy", "pool", "window_k", "window", "eta", "alpha", "margin",
              "threshold", "misses", "switches"]
CAND = ("lru", "clock", "lru2_visit", "arc_visit", "lirs_visit")
KEY = ["trace", "switcher", "policy", "pool", "window_k", "eta", "alpha"]
WINDOW_OF = {16: 36000, 4: 144000, 8: 72000, 32: 18000, 64: 9000}


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_sf(root, sf="0.01", hind=None, fams=None, free=None, traces=TRACES, windows=True,
             fixed_best=1000, valid=True, sim_misses=5, pools=POOLS):
    """Per-trace misses: every candidate 1100 except lru at fixed_best; belady 800."""
    sf_dir = root / "out" / f"sf{sf}"
    sf_dir.mkdir(parents=True, exist_ok=True)
    hind = hind if hind is not None else {16: {p: 1000 for p in pools.values()}}
    fams = fams if fams is not None else []
    free = free if free is not None else hind
    ks = sorted(set(hind) | set(free) | {f[1] for f in fams})
    for t in traces:
        res, ph, occ, lay = [], [], [], []
        for pool in pools.values():
            for c in CAND:
                m = fixed_best if c == "lru" else 1100
                res.append([t, "fixed", c, pool, "", "", "", "", "", "", m, ""])
                for phase in range(4):
                    ph.append([t, "fixed", c, pool, "", "", "", phase, "comp", m // 4])
            res.append([t, "belady", "belady", pool, "", "", "", "", "", "", 800, ""])
            for k in ks:
                w = WINDOW_OF[k]
                if k in free:
                    res.append([t, "free_bound", "", pool, k, w, "", "", "", "", free[k][pool], ""])
                if k in hind:
                    res.append([t, "hindsight", "", pool, k, w, "", "", "", "", hind[k][pool], 3])
                    for phase in range(4):
                        ph.append([t, "hindsight", "", pool, k, "", "", phase, "comp", hind[k][pool] // 4])
                        occ.append([t, "hindsight", "", pool, k, "", "", phase, "lru", 100])
                        occ.append([t, "hindsight", "", pool, k, "", "", phase, "arc_visit", 50])
            for (fam, k, eta, alpha, m, sw) in fams:
                w = WINDOW_OF[k]
                e = "" if eta is None else eta
                a = "" if alpha is None else alpha
                mg = "" if fam == "two_window" else 2
                th = 0.05 if fam == "two_window" else ""
                res.append([t, fam, "", pool, k, w, e, a, mg, th, m[pool], sw])
                for phase in range(4):
                    ph.append([t, fam, "", pool, k, e, a, phase, "comp", m[pool] // 4])
                    occ.append([t, fam, "", pool, k, e, a, phase, "lru", 100])
        write_csv(sf_dir / f"{t}_results.csv", RES_HEADER, res)
        write_csv(sf_dir / f"{t}_phases.csv", KEY + ["phase", "component", "misses"], ph)
        write_csv(sf_dir / f"{t}_occupancy.csv", KEY + ["phase", "candidate", "runs"], occ)
        write_csv(sf_dir / f"{t}_switches.csv", KEY + ["run", "from", "to"], [])
        write_csv(sf_dir / f"{t}_layout.csv", ["trace", "phase", "component", "first_run", "runs"],
                  [[t, 0, "comp", 0, 100]])
        if valid:
            write_csv(sf_dir / f"{t}_validation.csv", ["trace", "policy", "pool", "pass", "warmup", "misses"],
                      [["index_stream_s1", "lru", 145, 0, 1, 5]])
    if windows:
        write_csv(sf_dir / "windows.csv", ["window_k", "window", "mean_phase_runs"],
                  [[k, WINDOW_OF[k], 1000] for k in WINDOW_OF])
    sim = root / "sim" / f"sf{sf}"
    sim.mkdir(parents=True, exist_ok=True)
    write_csv(sim / "index_stream_s1_passes.csv",
              ["trace", "policy", "pool", "pass", "warmup", "items", "fetches", "runs", "misses",
               "query_misses", "lookup_misses"],
              [["index_stream_s1", "lru", 145, 0, 1, 1, 1, 1, sim_misses, 0, 0]])
    return sf_dir


def fam(name, k, eta, alpha, m25, m10=1000, m50=1000, sw=2):
    return (name, k, eta, alpha, {145: m10, 362: m25, 724: m50}, sw)


def run_main(root, sf="0.01"):
    return mod.main(["--sf", sf, "--out", str(root / "out"), "--sim-dir", str(root / "sim")])


def summed_for(root, sf="0.01"):
    return mod.summed(mod.read_results(root / "out" / f"sf{sf}"))


def picks_for(window_k=16):
    return {
        "two_window": dict(window_k=window_k, eta=None, alpha=None, margin=None, threshold=0.05, misses=0, switches=0),
        "hedge": dict(window_k=window_k, eta=10.0, alpha=0.0, margin=2.0, threshold=None, misses=0, switches=0),
        "fixed_share": dict(window_k=window_k, eta=10.0, alpha=0.1, margin=2.0, threshold=None, misses=0, switches=0),
    }


def all_fams(m25, m10=1000, m50=1000):
    return [fam("two_window", 16, None, None, m25, m10, m50),
            fam("hedge", 16, 10, 0, m25, m10, m50),
            fam("fixed_share", 16, 10, 0.1, m25, m10, m50)]


def test_pick_lowest_misses_and_ties(tmp_path):
    hind = {16: {145: 1000, 362: 1000, 724: 1000}, 4: {145: 1000, 362: 1000, 724: 1000}}
    fams = [
        fam("hedge", 16, 10, 0, 900), fam("hedge", 8, 10, 0, 900), fam("hedge", 4, 20, 0, 900),
        fam("hedge", 16, 5, 0, 900),
        fam("hedge", 16, 20, 0, 899),
        fam("fixed_share", 16, 10, 0.1, 900), fam("fixed_share", 16, 10, 0.03, 900),
        fam("two_window", 16, None, None, 950), fam("two_window", 4, None, None, 950),
    ]
    write_sf(tmp_path, hind=hind, fams=fams, free=hind)
    s = summed_for(tmp_path)
    picks = mod.pick_settings(s, 362)
    assert picks["hedge"]["misses"] == 3 * 899 and picks["hedge"]["eta"] == 20
    # tie on misses and switches: k = 16 beats 8 and 4, then smaller alpha
    assert picks["two_window"]["window_k"] == 16
    assert picks["fixed_share"]["alpha"] == 0.03
    # fewer switches beats k order
    fams2 = [fam("hedge", 16, 10, 0, 900, sw=5), fam("hedge", 4, 10, 0, 900, sw=1)]
    write_sf(tmp_path / "b", hind=hind, fams=fams2, free=hind)
    p2 = mod.pick_settings(summed_for(tmp_path / "b"), 362)
    assert p2["hedge"]["window_k"] == 4
    # eta ties: smaller eta wins
    fams3 = [fam("hedge", 16, 20, 0, 900), fam("hedge", 16, 5, 0, 900)]
    write_sf(tmp_path / "c", hind=hind, fams=fams3, free=hind)
    assert mod.pick_settings(summed_for(tmp_path / "c"), 362)["hedge"]["eta"] == 5


@pytest.mark.parametrize("other", [8, 32, 4, 64])
def test_k16_wins_ties_over_each_k(tmp_path, other):
    hind = {16: {145: 1000, 362: 1000, 724: 1000}}
    write_sf(tmp_path, hind=hind, fams=[fam("hedge", other, 10, 0, 900), fam("hedge", 16, 10, 0, 900)], free=hind)
    assert mod.pick_settings(summed_for(tmp_path), 362)["hedge"]["window_k"] == 16


def test_pick_ignores_settings_missing_a_trace(tmp_path):
    hind = {16: {145: 1000, 362: 1000, 724: 1000}}
    write_sf(tmp_path, hind=hind, fams=[fam("hedge", 16, 10, 0, 900)], free=hind, traces=TRACES[:2])
    s = mod.summed(mod.read_results(tmp_path / "out" / "sf0.01", allow_missing=True))
    assert "hedge" not in mod.pick_settings(s, 362)


def test_chosen_roundtrip(tmp_path):
    p = tmp_path / "chosen.json"
    mod.write_chosen(p, picks_for())
    got = mod.read_chosen(p)
    assert got["hedge"]["eta"] == 10.0 and got["two_window"]["threshold"] == 0.05
    raw = json.loads(p.read_text())
    assert raw["sf"] == "0.01" and raw["bar_fixed_on"] == "2026-11-10" and "picked_on" in raw


def test_read_chosen_refuses_bad_files(tmp_path):
    p = tmp_path / "chosen.json"
    with pytest.raises(mod.ChosenError) as e:
        mod.read_chosen(p)
    assert str(p) in str(e.value)
    p.write_text("not json")
    with pytest.raises(mod.ChosenError):
        mod.read_chosen(p)
    mod.write_chosen(p, picks_for())
    raw = json.loads(p.read_text())
    raw["sf"] = "0.1"
    p.write_text(json.dumps(raw))
    with pytest.raises(mod.ChosenError):
        mod.read_chosen(p)
    raw["sf"] = "0.01"
    del raw["families"]["hedge"]
    p.write_text(json.dumps(raw))
    with pytest.raises(mod.ChosenError):
        mod.read_chosen(p)


def verdict_for(tmp_path, hind25, fam25, m10=1000, m50=1000, extra=None):
    hind = {16: {145: 1000, 362: hind25, 724: 1000}}
    fams = extra if extra is not None else [fam("fixed_share", 16, 10, 0.1, fam25, m10, m50)]
    write_sf(tmp_path, hind=hind, fams=fams, free=hind)
    s = summed_for(tmp_path)
    return mod.verdict(s, picks_for(), POOLS)


def test_verdict_go(tmp_path):
    v = verdict_for(tmp_path, 920, 950)
    assert v["go"] and v["handed_over"] == "fixed_share"
    f = v["families"]["fixed_share"]
    assert f["room_pass"] and f["keep_pass"] and f["harm_pass"]


def test_verdict_no_go_room(tmp_path):
    v = verdict_for(tmp_path, 951, 951)
    assert not v["go"] and v["handed_over"] is None
    assert not v["families"]["fixed_share"]["room_pass"]


def test_verdict_no_go_keep(tmp_path):
    v = verdict_for(tmp_path, 920, 961)
    f = v["families"]["fixed_share"]
    assert f["room_pass"] and not f["keep_pass"] and not v["go"]


def test_verdict_no_go_harm_10(tmp_path):
    v = verdict_for(tmp_path, 920, 950, m10=1012)
    f = v["families"]["fixed_share"]
    assert f["room_pass"] and f["keep_pass"] and not f["harm_pass"] and not v["go"]


def test_verdict_no_go_harm_50(tmp_path):
    v = verdict_for(tmp_path, 920, 950, m50=1011)
    assert not v["families"]["fixed_share"]["harm_pass"] and not v["go"]
    ok = verdict_for(tmp_path / "ok", 920, 950, m50=1010)
    assert ok["families"]["fixed_share"]["harm_pass"]


def test_handed_over_is_largest_saving(tmp_path):
    extra = [fam("hedge", 16, 10, 0, 930), fam("fixed_share", 16, 10, 0.1, 940)]
    v = verdict_for(tmp_path, 900, 0, extra=extra)
    assert v["families"]["hedge"]["passed"] and v["families"]["fixed_share"]["passed"]
    assert v["go"] and v["handed_over"] == "hedge"


def test_validation_mismatch_exits_1(tmp_path):
    write_sf(tmp_path, fams=all_fams(950), sim_misses=6)
    assert run_main(tmp_path) == 1
    report = (tmp_path / "out" / "sf0.01" / "report.md").read_text()
    assert report.startswith("**WARNING**")
    assert mod.check_validation(tmp_path / "out" / "sf0.01", tmp_path / "sim")


def test_validation_ok_exits_0(tmp_path):
    write_sf(tmp_path, fams=all_fams(950))
    assert mod.check_validation(tmp_path / "out" / "sf0.01", tmp_path / "sim") == []
    assert run_main(tmp_path) == 0


def test_analyze_refuses_missing_trace_or_window_drift(tmp_path):
    write_sf(tmp_path / "a", fams=all_fams(950), traces=TRACES[:2])
    with pytest.raises(ValueError):
        mod.read_results(tmp_path / "a" / "out" / "sf0.01")
    assert run_main(tmp_path / "a") == 1
    assert (tmp_path / "a" / "out" / "sf0.01" / "report.md").read_text().startswith("**WARNING**")
    write_sf(tmp_path / "b", fams=all_fams(950), windows=False)
    assert run_main(tmp_path / "b") == 1
    assert (tmp_path / "b" / "out" / "sf0.01" / "report.md").read_text().startswith("**WARNING**")
    write_sf(tmp_path / "c", fams=all_fams(950))
    w = tmp_path / "c" / "out" / "sf0.01" / "windows.csv"
    write_csv(w, ["window_k", "window", "mean_phase_runs"], [[16, 99999, 1000], [4, 144000, 1000]])
    assert run_main(tmp_path / "c") == 1
    assert (tmp_path / "c" / "out" / "sf0.01" / "report.md").read_text().startswith("**WARNING**")


def test_sf01_writes_chosen_and_dev_heading(tmp_path):
    write_sf(tmp_path, fams=all_fams(950))
    assert run_main(tmp_path) == 0
    chosen = json.loads((tmp_path / "out" / "chosen.json").read_text())
    assert set(chosen["families"]) == set(mod.FAMILIES)
    report = (tmp_path / "out" / "sf0.01" / "report.md").read_text()
    assert "development run, not a decision" in report
    assert (tmp_path / "out" / "sf0.01" / "cells.csv").exists()
    assert (tmp_path / "out" / "sf0.01" / "misses_sf0.01.png").exists()


def test_sf1_needs_chosen_then_decides(tmp_path):
    pools = {0.1: 1442, 0.25: 3604, 0.5: 7208}
    big = {1442: 1000, 3604: 920, 7208: 1000}
    m = {1442: 1000, 3604: 950, 7208: 1000}
    fams = [("two_window", 16, None, None, m, 1), ("hedge", 16, 10, 0, m, 1), ("fixed_share", 16, 10, 0.1, m, 1)]
    write_sf(tmp_path, sf="0.1", hind={16: big}, fams=fams, free={16: big}, pools=pools)
    assert run_main(tmp_path, "0.1") == 1
    r = (tmp_path / "out" / "sf0.1" / "report.md").read_text()
    assert r.startswith("**WARNING**") and "**NO VERDICT**" in r
    mod.write_chosen(tmp_path / "out" / "chosen.json", picks_for())
    assert run_main(tmp_path, "0.1") == 0
    r = (tmp_path / "out" / "sf0.1" / "report.md").read_text()
    assert r.startswith("**GO**")


def test_clean_run_writes_chosen_and_warning_run_removes_it(tmp_path):
    write_sf(tmp_path, fams=all_fams(950))
    chosen = tmp_path / "out" / "chosen.json"
    assert run_main(tmp_path) == 0 and chosen.exists()
    write_sf(tmp_path / "bad", fams=all_fams(950), sim_misses=6)
    (tmp_path / "bad" / "out" / "chosen.json").write_text("{}")
    assert run_main(tmp_path / "bad") == 1
    assert (tmp_path / "bad" / "out" / "chosen.json").read_text() == "{}"
    report = (tmp_path / "bad" / "out" / "sf0.01" / "report.md").read_text()
    assert "chosen.json was not written" in report and "left as it is" in report
    write_sf(tmp_path / "short", fams=all_fams(950), traces=TRACES[:2])
    assert run_main(tmp_path / "short") == 1
    assert not (tmp_path / "short" / "out" / "chosen.json").exists()


def test_warned_run_leaves_a_valid_frozen_file_alone(tmp_path):
    write_sf(tmp_path, fams=all_fams(950), sim_misses=6)
    frozen = tmp_path / "out" / "chosen.json"
    mod.write_chosen(frozen, picks_for())
    before = frozen.read_bytes()
    assert run_main(tmp_path) == 1
    assert frozen.read_bytes() == before


def test_identical_rerun_keeps_frozen_file_byte_identical(tmp_path):
    write_sf(tmp_path, fams=all_fams(950))
    frozen = tmp_path / "out" / "chosen.json"
    assert run_main(tmp_path) == 0
    doc = json.loads(frozen.read_text())
    doc["picked_on"] = "2020-01-01"
    frozen.write_text(json.dumps(doc, indent=2) + "\n")
    before = frozen.read_bytes()
    assert run_main(tmp_path) == 0
    assert frozen.read_bytes() == before


def test_changed_picks_refuse_and_keep_the_old_file(tmp_path):
    write_sf(tmp_path, fams=all_fams(950))
    assert run_main(tmp_path) == 0
    frozen = tmp_path / "out" / "chosen.json"
    doc = json.loads(frozen.read_text())
    doc["families"]["hedge"]["eta"] = 99.0
    frozen.write_text(json.dumps(doc, indent=2) + "\n")
    before = frozen.read_bytes()
    assert run_main(tmp_path) == 1
    assert frozen.read_bytes() == before
    report = (tmp_path / "out" / "sf0.01" / "report.md").read_text()
    assert report.startswith("**WARNING**") and "picks changed for hedge" in report
    assert "Delete chosen.json by hand" in report


def test_hand_over_line_names_the_setting(tmp_path):
    extra = [fam("two_window", 16, None, None, 1000), fam("hedge", 16, 10, 0, 930), fam("fixed_share", 16, 10, 0.1, 940)]
    v = verdict_for(tmp_path, 900, 0, extra=extra)
    lines = mod.verdict_lines(v, picks_for(), POOLS, dev=False)
    assert "Handed over: hedge (k=16, eta=10, alpha=0)" in lines


def test_report_gives_each_window_in_runs_and_cells_has_window(tmp_path):
    import pandas as pd
    write_sf(tmp_path, fams=all_fams(950))
    assert run_main(tmp_path) == 0
    sf_dir = tmp_path / "out" / "sf0.01"
    report = (sf_dir / "report.md").read_text()
    assert "## Windows" in report and "| 16 | 36,000 |" in report and "| 64 | 9,000 |" in report
    cells = pd.read_csv(sf_dir / "cells.csv")
    assert "window" in cells.columns
    hind = cells[(cells["switcher"] == "hindsight") & (cells["window_k"] == 16)]
    assert set(hind["window"]) == {36000}
    assert cells[cells["switcher"] == "fixed"]["window"].isna().all()


def test_verdict_under_warnings_is_marked_not_valid(tmp_path):
    pools = {0.1: 1442, 0.25: 3604, 0.5: 7208}
    big = {1442: 1000, 3604: 920, 7208: 1000}
    m = {1442: 1000, 3604: 950, 7208: 1000}
    fams = [("two_window", 16, None, None, m, 1), ("hedge", 16, 10, 0, m, 1), ("fixed_share", 16, 10, 0.1, m, 1)]
    write_sf(tmp_path, sf="0.1", hind={16: big}, fams=fams, free={16: big}, pools=pools, sim_misses=6)
    mod.write_chosen(tmp_path / "out" / "chosen.json", picks_for())
    assert run_main(tmp_path, "0.1") == 1
    r = (tmp_path / "out" / "sf0.1" / "report.md").read_text()
    assert r.startswith("**WARNING**")
    assert "**GO** (not valid: see warnings)" in r or "**NO-GO** (not valid: see warnings)" in r


def test_missing_or_empty_validation_file_is_a_message(tmp_path):
    sf_dir = write_sf(tmp_path, fams=all_fams(950))
    (sf_dir / "phases_s2_validation.csv").unlink()
    (sf_dir / "phases_s3_validation.csv").write_text("trace,policy,pool,pass,warmup,misses\n")
    msgs = mod.check_validation(sf_dir, tmp_path / "sim")
    assert any("phases_s2_validation.csv" in m for m in msgs)
    assert any("phases_s3_validation.csv" in m for m in msgs)
    assert run_main(tmp_path) == 1


def test_sf1_refuses_chosen_with_other_margin_or_threshold(tmp_path):
    pools = {0.1: 1442, 0.25: 3604, 0.5: 7208}
    big = {1442: 1000, 3604: 920, 7208: 1000}
    m = {1442: 1000, 3604: 950, 7208: 1000}
    fams = [("two_window", 16, None, None, m, 1), ("hedge", 16, 10, 0, m, 1), ("fixed_share", 16, 10, 0.1, m, 1)]
    write_sf(tmp_path, sf="0.1", hind={16: big}, fams=fams, free={16: big}, pools=pools)
    picks = picks_for()
    picks["hedge"]["margin"] = 3.0
    mod.write_chosen(tmp_path / "out" / "chosen.json", picks)
    assert run_main(tmp_path, "0.1") == 1
    r = (tmp_path / "out" / "sf0.1" / "report.md").read_text()
    assert r.startswith("**WARNING**") and "hedge" in r.split("\n\n")[1]
    assert "**NO VERDICT**" in r and "**GO**" not in r and "**NO-GO**" not in r
    picks = picks_for()
    picks["two_window"]["threshold"] = 0.1
    mod.write_chosen(tmp_path / "out" / "chosen.json", picks)
    assert run_main(tmp_path, "0.1") == 1
    r = (tmp_path / "out" / "sf0.1" / "report.md").read_text()
    assert "**NO VERDICT**" in r and "**GO**" not in r and "**NO-GO**" not in r


@pytest.mark.parametrize("fam,field,value", [("hedge", "eta", None), ("fixed_share", "alpha", "x"),
                                             ("two_window", "threshold", True), ("hedge", "margin", "drop")])
def test_read_chosen_refuses_a_setting_without_its_numbers(tmp_path, fam, field, value):
    picks = picks_for()
    if value == "drop":
        del picks[fam][field]
    else:
        picks[fam][field] = value
    mod.write_chosen(tmp_path / "chosen.json", picks)
    with pytest.raises(mod.ChosenError, match=f"{fam} setting has no usable {field}"):
        mod.read_chosen(tmp_path / "chosen.json")
