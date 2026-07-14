"""Pure metric functions implementing Baseline Results Protocol v1.0."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd

ANNUALIZATION = 252
PORTFOLIO_COLUMNS = ("AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar")
RANKING_COLUMNS = ("IC", "ICIR", "RankIC", "RankICIR")


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def ranking_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Compute daily cross-sectional Pearson/Spearman means and IRs."""
    if not {"score", "label"}.issubset(frame.columns):
        raise ValueError("ranking frame must contain score and label columns")
    if not isinstance(frame.index, pd.MultiIndex) or "datetime" not in frame.index.names:
        raise ValueError("ranking frame must have a datetime/instrument MultiIndex")

    clean = frame[["score", "label"]].replace([np.inf, -np.inf], np.nan).dropna()
    daily_ic = clean.groupby(level="datetime", sort=True).apply(
        lambda x: x["score"].corr(x["label"]), include_groups=False
    ).dropna()
    daily_rank_ic = clean.groupby(level="datetime", sort=True).apply(
        lambda x: x["score"].corr(x["label"], method="spearman"), include_groups=False
    ).dropna()

    def summarize(values: pd.Series) -> tuple[float, float]:
        mean = float(values.mean()) if len(values) else float("nan")
        std = float(values.std(ddof=1)) if len(values) >= 2 else float("nan")
        return mean, _safe_ratio(mean, std)

    ic, icir = summarize(daily_ic)
    rank_ic, rank_icir = summarize(daily_rank_ic)
    return {"IC": ic, "ICIR": icir, "RankIC": rank_ic, "RankICIR": rank_icir}


def portfolio_metrics(daily_ret_net: pd.Series | np.ndarray) -> dict[str, float | int]:
    """Compute protocol portfolio metrics from daily net simple returns."""
    returns = pd.Series(daily_ret_net, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if (returns <= -1).any():
        bad = returns[returns <= -1].iloc[0]
        raise ValueError(f"daily_ret_net must be greater than -1; got {bad}")
    g = np.log1p(returns.to_numpy())
    n = len(g)
    mean_g = float(np.mean(g)) if n else float("nan")
    std_g = float(np.std(g, ddof=1)) if n >= 2 else float("nan")
    ar = float(np.exp(mean_g * ANNUALIZATION) - 1) if n else float("nan")
    std = float(std_g * math.sqrt(ANNUALIZATION))
    nav = np.concatenate(([1.0], np.exp(np.cumsum(g))))
    mdd = float(np.min(nav / np.maximum.accumulate(nav) - 1)) if n else float("nan")
    sharpe = _safe_ratio(math.sqrt(ANNUALIZATION) * mean_g, std_g)
    downside = float(np.sqrt(np.mean(np.minimum(g, 0.0) ** 2))) if n else float("nan")
    sortino = _safe_ratio(math.sqrt(ANNUALIZATION) * mean_g, downside)
    calmar = _safe_ratio(ar, abs(mdd))
    return {
        "AR": ar,
        "STD": std,
        "MDD": mdd,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "num_test_days": n,
    }


def aggregate_seed_metrics(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate numeric protocol metrics across seeds with ddof=1."""
    metrics = list(RANKING_COLUMNS + PORTFOLIO_COLUMNS)
    rows = []
    for (market, model), group in seed_metrics.groupby(["market", "model"], sort=True):
        row: dict[str, object] = {"market": market, "model": model}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    columns = ["market", "model"] + [f"{m}_{s}" for m in metrics for s in ("mean", "std")]
    return pd.DataFrame(rows, columns=columns)


def mean_std_table(aggregate: pd.DataFrame) -> pd.DataFrame:
    """Create a four-decimal display table; never use it as a metric source."""
    rows = []
    for _, source in aggregate.iterrows():
        row = {"market": source["market"], "model": source["model"]}
        for metric in RANKING_COLUMNS + PORTFOLIO_COLUMNS:
            row[metric] = f"{source[f'{metric}_mean']:.4f} ± {source[f'{metric}_std']:.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def metrics_from_curve(curve: pd.DataFrame) -> Mapping[str, float | int]:
    return portfolio_metrics(curve["daily_ret_net"])
