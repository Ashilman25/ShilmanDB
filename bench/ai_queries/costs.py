#Cost weights: what a row costs on each cheaper tier, with the big model at 1
from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class Weights:
    small: float
    proxy: float
    embed: float


@dataclass
class Usage:
    big: int = 0
    small: int = 0
    proxy: int = 0
    embed_rows: float = 0.0

    #Cost in big-call equivalents
    def cost(self, w: Weights) -> float:
        return self.big + self.small * w.small + self.proxy * w.proxy + self.embed_rows * w.embed


def _row(df: pd.DataFrame, column: str, value: str, what: str) -> pd.Series:
    rows = df[df[column] == value]
    if rows.empty:
        raise ValueError(f"{what}: no row for {value!r}")
    return rows.iloc[0]


#Rows-weighted ms, input tokens and output tokens per row over `qids`
def _tier_per_row(latency: pd.DataFrame, tier: str, qids: list[str]) -> tuple[float, float, float]:
    t = latency[(latency["tier"] == tier) & latency["qid"].isin(qids)]
    rows = t["rows"].to_numpy(dtype=float)
    avg = lambda col: float((t[col].to_numpy(dtype=float) * rows).sum() / rows.sum())  # noqa: E731
    return avg("median_ms"), avg("mean_in_tokens"), avg("mean_out_tokens")


def _price(row: pd.Series, tokens_in: float, tokens_out: float) -> float:
    return (tokens_in * float(row["usd_per_m_in"]) + tokens_out * float(row["usd_per_m_out"])) / 1e6


#Each weight is the larger of its time ratio and its price ratio against the big model.
def weights_from(latency: pd.DataFrame, prices: pd.DataFrame, embed_timing: pd.DataFrame, proxy_us_per_row: float) -> Weights:
    by_tier = {t: set(latency.loc[latency["tier"] == t, "qid"]) for t in ("big", "small")}
    for tier, qids in by_tier.items():
        if not qids:
            raise ValueError(f"latency: no rows for tier {tier!r}")

    shared = sorted(by_tier["big"] & by_tier["small"])
    if not shared:
        raise ValueError("latency: big and small share no question")
    
    big_ms, big_in, big_out = _tier_per_row(latency, "big", shared)
    small_ms, small_in, small_out = _tier_per_row(latency, "small", shared)
    big_usd = _price(_row(prices, "tier", "big", "prices"), big_in, big_out)
    small_usd = _price(_row(prices, "tier", "small", "prices"), small_in, small_out)
    small = max(small_ms / big_ms, small_usd / big_usd)

    proxy = proxy_us_per_row / 1000.0 / big_ms  # runs on the local CPU: time only

    timing = _row(embed_timing, "pass", "test", "embed timing")
    embed_ms = float(timing["seconds"]) / float(timing["rows"]) * 1000.0
    embed_usd = _price(_row(prices, "tier", "embed", "prices"), big_in, 0.0)
    embed = max(embed_ms / big_ms, embed_usd / big_usd)
    return Weights(small=small, proxy=proxy, embed=embed)
