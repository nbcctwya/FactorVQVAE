"""Qlib 0.9.7 standard TopK-DropN backtest adapter."""

from __future__ import annotations

import pandas as pd
import qlib
from qlib.backtest import backtest

STRATEGY_KWARGS = {
    "topk": 30,
    "n_drop": 5,
    "method_sell": "bottom",
    "method_buy": "top",
    "hold_thresh": 1,
    "only_tradable": False,
    "forbid_all_trade_at_limit": True,
    "risk_degree": 0.95,
}
ACCOUNT = 100_000_000
EXCHANGE_KWARGS = {
    "freq": "day",
    "open_cost": 0.0005,
    "close_cost": 0.0015,
    "min_cost": 0.0,
    "deal_price": "close",
}


def init_market(provider_uri: str, region: str) -> None:
    qlib.init(provider_uri=provider_uri, region=region)


def expected_calendar(start_time: str, end_time: str) -> pd.DatetimeIndex:
    from qlib.data import D

    return pd.DatetimeIndex(D.calendar(start_time=start_time, end_time=end_time, freq="day"))


def validate_signal_coverage(signal: pd.Series, start_time: str, end_time: str) -> None:
    if not isinstance(signal.index, pd.MultiIndex) or signal.index.names != ["datetime", "instrument"]:
        raise ValueError("signal index names/order must be datetime, instrument")
    if signal.index.has_duplicates:
        raise ValueError("signal contains duplicate datetime/instrument keys")
    dates = pd.DatetimeIndex(signal.index.get_level_values("datetime").unique()).sort_values()
    calendar = expected_calendar(start_time, end_time)
    missing = calendar.difference(dates)
    extra = dates.difference(calendar)
    if len(missing) or len(extra):
        raise ValueError(
            f"prediction coverage differs from test calendar: missing={missing[:5].tolist()}, "
            f"extra={extra[:5].tolist()}"
        )


def run_standard_backtest(
    signal: pd.Series,
    *,
    start_time: str,
    end_time: str,
    benchmark: str,
    deal_price: str = "close",
    limit_threshold: float | None = None,
) -> pd.DataFrame:
    """Run the single canonical path used by both seed and ensemble signals."""
    validate_signal_coverage(signal, start_time, end_time)
    strategy = {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy.signal_strategy",
        "kwargs": {"signal": signal.sort_index(), **STRATEGY_KWARGS},
    }
    executor = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
            "verbose": False,
        },
    }
    exchange = {
        **EXCHANGE_KWARGS,
        "deal_price": deal_price,
        "limit_threshold": limit_threshold,
    }
    portfolio, _ = backtest(
        start_time=start_time,
        end_time=end_time,
        strategy=strategy,
        executor=executor,
        benchmark=benchmark,
        account=ACCOUNT,
        exchange_kwargs=exchange,
    )
    report = portfolio["1day"][0].copy()
    required = {"return", "cost", "bench"}
    if not required.issubset(report.columns):
        raise RuntimeError(f"Qlib report missing columns: {sorted(required - set(report.columns))}")
    # Qlib 0.9.7 Account.update_portfolio_metrics records return_rate as
    # (now_earning + now_cost) / last_account_value, i.e. gross of costs.
    curve = pd.DataFrame(index=pd.DatetimeIndex(report.index))
    curve.index.name = "datetime"
    curve["daily_ret_gross"] = report["return"].astype(float)
    curve["cost"] = report["cost"].astype(float)
    curve["daily_ret_net"] = curve["daily_ret_gross"] - curve["cost"]
    curve["bench_ret"] = report["bench"].astype(float)
    curve["nav"] = (1.0 + curve["daily_ret_net"]).cumprod()
    curve["bench_nav"] = (1.0 + curve["bench_ret"]).cumprod()
    if not len(curve) or not curve.index.is_monotonic_increasing or curve.index.has_duplicates:
        raise RuntimeError("Qlib returned an invalid daily curve")
    if not pd.notna(curve).all().all():
        raise RuntimeError("Qlib returned NaN in the daily curve")
    return curve.reset_index()
