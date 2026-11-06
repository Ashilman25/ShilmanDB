#Turn the raw Amazon review files into the test table and the 500K slice.
import argparse
import gzip
import hashlib
import heapq
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from .common import RESULTS, paths, text_key, write_text_atomic

REVIEW_COLUMNS = ["review_id", "rating", "title", "text", "parent_asin", "timestamp", "helpful_vote", "verified_purchase", "user_id", "text_key"]
PRODUCT_COLUMNS = ["parent_asin", "title", "store", "price", "main_category", "categories", "average_rating", "rating_number"]

_PRICE = re.compile(r"^\$?\s*(\d{1,3}(,\d{3})+|\d+)(\.\d+)?$")
_CASE = re.compile(r"\bcases?\b", re.IGNORECASE)


def _open(path: Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


#Every line with its 0-based number
# None for a blank or bad line
def _numbered(path: Path) -> Iterator[tuple[int, dict | None]]:
    with _open(path) as f:
        for n, line in enumerate(f):
            try:
                rec = json.loads(line)
            except ValueError:
                rec = None
            yield n, rec if isinstance(rec, dict) else None


def read_jsonl(path: Path) -> Iterator[dict]:
    for _, rec in _numbered(path):
        if rec is not None:
            yield rec


#A plain number or a '$12.99' style string; anything else is NaN
def parse_price(v) -> float:
    if isinstance(v, bool):
        return math.nan
    
    if isinstance(v, (int, float)):
        return float(v) if math.isfinite(v) else math.nan
    
    if isinstance(v, str) and _PRICE.match(v.strip()):
        return float(v.strip().lstrip("$").replace(",", ""))
    
    return math.nan


def is_case(categories) -> bool:
    if isinstance(categories, str):
        try:
            categories = json.loads(categories)
        except ValueError:
            categories = [categories]
            
    if not isinstance(categories, list):
        return False
    return any(isinstance(c, str) and _CASE.search(c) for c in categories)


#A fixed number for (seed, review): the same on every run and machine
def _key(seed: int, review_id: int, tag: str = "") -> int:
    digest = hashlib.sha1(f"{seed}:{tag}{review_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _count(path: Path, exclude: frozenset[int] = frozenset()) -> tuple[dict[str, int], int]:
    counts: Counter[str] = Counter()
    skipped = 0
    for n, rec in _numbered(path):
        if rec is None:
            skipped += 1
        elif n not in exclude and isinstance(rec.get("parent_asin"), str):
            counts[rec["parent_asin"]] += 1
            
    return dict(counts), skipped


def count_reviews(path: Path) -> dict[str, int]:
    return _count(path)[0]


#Seeded shuffle of the big-enough products, added until the target is reached
def select_products(counts: dict[str, int], eligible: set[str], floor: int, cap: int, target_rows: int, seed: int) -> list[str]:
    pool = sorted(p for p, c in counts.items() if c >= floor and p in eligible)
    order = np.random.default_rng(seed).permutation(len(pool))
    chosen, total = [], 0
    
    for i in order:
        if total >= target_rows:
            break
        chosen.append(pool[i])
        total += min(counts[pool[i]], cap)
    return chosen


def _review_row(review_id: int, rec: dict) -> dict:
    title, text = rec.get("title") or "", rec.get("text") or ""
    rating = rec.get("rating")
    
    if isinstance(rating, float) and rating.is_integer():
        rating = int(rating)
        
    return {
        "review_id": review_id, "rating": rating, "title": title, "text": text,
        "parent_asin": rec["parent_asin"], "timestamp": rec.get("timestamp"),
        "helpful_vote": rec.get("helpful_vote"), "verified_purchase": rec.get("verified_purchase"),
        "user_id": rec.get("user_id"), "text_key": text_key(title, text)
    }


def _product_row(rec: dict) -> dict:
    price = parse_price(rec.get("price"))
    cats = rec.get("categories")
    
    return {
        "parent_asin": rec["parent_asin"], "title": rec.get("title") or "",
        "store": rec.get("store") or "", "price": "" if math.isnan(price) else price,
        "main_category": rec.get("main_category") or "",
        "categories": json.dumps(cats if isinstance(cats, list) else []),
        "average_rating": rec.get("average_rating"), "rating_number": rec.get("rating_number")
    }


def _meta_asins(meta_path: Path) -> set[str]:
    return {r["parent_asin"] for r in read_jsonl(meta_path) if isinstance(r.get("parent_asin"), str)}


def _write_products(meta_path: Path, wanted: set[str], out: Path) -> int:
    seen: dict[str, dict] = {}
    for rec in read_jsonl(meta_path):
        asin = rec.get("parent_asin")
        if asin in wanted and asin not in seen:
            seen[asin] = _product_row(rec)
            
    rows = [seen[a] for a in sorted(seen)]
    pd.DataFrame(rows, columns=PRODUCT_COLUMNS).to_csv(out, index=False, lineterminator="\n")
    return len(rows)


def _write_reviews(rows: list[dict], out: Path, seed: int) -> None:
    rows.sort(key=lambda r: _key(seed, r["review_id"], "order:"))
    pd.DataFrame(rows, columns=REVIEW_COLUMNS).to_csv(out, index=False, lineterminator="\n")


def build_test_table(reviews_path: Path, meta_path: Path, out_dir: Path, floor: int = 50, cap: int = 300, target_rows: int = 30000, seed: int = 1, exclude_ids: frozenset[int] = frozenset()) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts, skipped = _count(reviews_path, exclude_ids)
    chosen = set(select_products(counts, _meta_asins(meta_path), floor, cap, target_rows, seed))

    # keep the `cap` reviews with the smallest keys for each chosen product
    heaps: dict[str, list] = {p: [] for p in chosen}
    for n, rec in _numbered(reviews_path):
        if rec is None or n in exclude_ids or rec.get("parent_asin") not in chosen:
            continue
        
        item = (-_key(seed, n), n, _review_row(n, rec))
        heap = heaps[rec["parent_asin"]]
        if len(heap) < cap:
            heapq.heappush(heap, item)
        elif item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)
            
    rows = [item[2] for heap in heaps.values() for item in heap]
    _write_reviews(rows, out_dir / "reviews.csv", seed)
    products = _write_products(meta_path, chosen, out_dir / "products.csv")

    design = {
        "floor": floor, "cap": cap, "target_rows": target_rows, "seed": seed, "rows": len(rows),
        "products": products, "reviews_file": Path(reviews_path).name,
        "meta_file": Path(meta_path).name, "skipped_lines": skipped
    }
    write_text_atomic(out_dir / "design.json", json.dumps(design, indent=2))
    return design


def build_slice(reviews_path: Path, meta_path: Path, out_dir: Path, n: int = 500_000, seed: int = 1) -> dict:
    out = Path(out_dir) / "slice"
    out.mkdir(parents=True, exist_ok=True)
    valid = np.array([i for i, rec in _numbered(reviews_path) if rec is not None and isinstance(rec.get("parent_asin"), str)], dtype=np.int64)
    if n < len(valid):
        valid = np.sort(np.random.default_rng(seed).choice(valid, size=n, replace=False))
        
    wanted_ids = set(valid.tolist())
    rows = [_review_row(i, rec) for i, rec in _numbered(reviews_path) if i in wanted_ids and rec is not None]
    _write_reviews(rows, out / "reviews.csv", seed)
    _write_products(meta_path, {r["parent_asin"] for r in rows}, out / "products.csv")

    seen: set[str] = set()
    dups = 0
    for r in rows:  # rows are now in the written order
        dups += r["text_key"] in seen
        seen.add(r["text_key"])
        
    info = {
        "rows": len(rows), 
        "duplicate_share": dups / len(rows) if rows else 0.0
    }
    write_text_atomic(out / "slice_info.json", json.dumps(info, indent=2))
    return info

# Read reviews.csv and products.csv with the right types
def read_tables(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    reviews = pd.read_csv(data_dir / "reviews.csv", dtype=str, keep_default_na=False)
    for col in ("review_id", "rating", "timestamp", "helpful_vote"):
        reviews[col] = pd.to_numeric(reviews[col]).astype("int64")
    reviews["verified_purchase"] = reviews["verified_purchase"] == "True"

    products = pd.read_csv(data_dir / "products.csv", dtype=str, keep_default_na=False)
    products["price"] = products["price"].map(parse_price).astype("float64")
    products["average_rating"] = pd.to_numeric(products["average_rating"].replace("", np.nan)).astype("float64")
    products["rating_number"] = pd.to_numeric(products["rating_number"].replace("", np.nan)).fillna(0).astype("int64")
    return reviews, products


MIN_TEST_ROWS = 20_000  # the real test table never goes below this


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "slice"):
        p = sub.add_parser(name)
        p.add_argument("--reviews", type=Path, required=True)
        p.add_argument("--meta", type=Path, required=True)
        p.add_argument("--seed", type=int, default=1)
        p.add_argument("--root", type=Path, default=RESULTS)
        if name == "build":
            p.add_argument("--floor", type=int, default=50)
            p.add_argument("--cap", type=int, default=300)
            p.add_argument("--rows", type=int, default=30000)
            p.add_argument("--exclude-from", type=Path, default=None)
        else:
            p.add_argument("--n", type=int, default=500_000)
            
    args = ap.parse_args(argv)
    data = paths(args.root).data
    if args.cmd == "build":
        if args.rows < MIN_TEST_ROWS:
            print(f"warning: --rows {args.rows} is below {MIN_TEST_ROWS:,}, the floor for the real test table; fine for a smoke run, not for the check", file=sys.stderr)
        exclude = frozenset()
        
        if args.exclude_from:
            ids = pd.read_csv(paths(args.exclude_from).data / "reviews.csv", usecols=["review_id"])
            exclude = frozenset(ids["review_id"].tolist())
        print(json.dumps(build_test_table(args.reviews, args.meta, data, args.floor, args.cap, args.rows, args.seed, exclude), indent=2))
    else:
        print(json.dumps(build_slice(args.reviews, args.meta, data, args.n, args.seed), indent=2))


if __name__ == "__main__":
    main()
