"""Generate Baseline Results Protocol v1.0 artifacts from existing predictions."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import re
import subprocess

import numpy as np
import pandas as pd
import qlib
import yaml

from baseline_results.metrics import (
    PORTFOLIO_COLUMNS,
    RANKING_COLUMNS,
    aggregate_seed_metrics,
    mean_std_table,
    portfolio_metrics,
    ranking_metrics,
)
from baseline_results.qlib_adapter import (
    ACCOUNT,
    EXCHANGE_KWARGS,
    STRATEGY_KWARGS,
    expected_calendar,
    init_market,
    run_standard_backtest,
)

BASELINE_ID = "factorvqvae"
SEED_COLUMNS = [
    "market", "model", "seed", *RANKING_COLUMNS, *PORTFOLIO_COLUMNS,
    "num_test_days", "pred_path_or_ckpt_path",
]
ENSEMBLE_COLUMNS = [
    "market", "model", "ensemble_method", *RANKING_COLUMNS, *PORTFOLIO_COLUMNS,
    "num_test_days", "seeds", "pred_paths",
]


def load_config(path: Path) -> dict:
    with path.open() as handle:
        config = yaml.safe_load(handle)
    for key in ("benchmark", "deal_price", "limit_threshold"):
        if key not in config.get("evaluation", {}):
            raise ValueError(f"{path}: evaluation.{key} must be explicitly configured")
    return config


def prediction_path(config: dict, seed: int) -> Path:
    market = config["experiment"]["market"]
    vq = config["vqvae"]["num_factors"]
    tf = config["transformer"]
    name = f"{market}_Stage2_VQ{vq}_Th{tf['hidden_size']}_h{tf['heads']}_l{tf['n_layers']}_sd{seed}.pkl"
    return Path(config["paths"]["result_dir"]) / name


def load_prediction(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"declared prediction is missing: {path}")
    frame = pd.read_pickle(path)
    if list(frame.columns) != ["score", "label"]:
        raise ValueError(f"{path}: expected exactly score,label columns")
    if frame.index.names != ["datetime", "instrument"] or frame.index.has_duplicates:
        raise ValueError(f"{path}: invalid prediction index")
    if not np.isfinite(frame[["score", "label"]].to_numpy()).all():
        raise ValueError(f"{path}: prediction contains NaN/Inf")
    return frame.sort_index()


def ensemble_prediction(frames: list[pd.DataFrame]) -> pd.DataFrame:
    joined = pd.concat(
        [frame.rename(columns={"score": f"score_{i}", "label": f"label_{i}"}) for i, frame in enumerate(frames)],
        axis=1,
        join="inner",
    )
    if joined.empty:
        raise ValueError("ensemble inner join is empty")
    labels = joined.filter(regex=r"^label_")
    if not labels.eq(labels.iloc[:, 0], axis=0).all().all():
        raise ValueError("seed labels disagree after ensemble alignment")
    scores = joined.filter(regex=r"^score_")
    return pd.DataFrame({"score": scores.mean(axis=1), "label": labels.iloc[:, 0]}, index=joined.index)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def market_metadata(config: dict) -> dict:
    data = config["data"]
    evaluation = config["evaluation"]
    region = data["region"]
    return {
        "provider_uri": data["provider_uri"],
        "region": region,
        "instruments": data["universe"],
        "instruments_membership": "Qlib dynamic historical constituents",
        "benchmark": evaluation["benchmark"],
        "deal_price": evaluation["deal_price"],
        "limit_threshold": evaluation["limit_threshold"],
        "suspension_and_untradable": "orders checked by Qlib Exchange; only_tradable=false affects candidate ranking only",
        "trade_unit": 100 if region == "cn" else 1,
        "shorting": False,
        "leverage": False,
    }


def build_eval_config(configs: list[dict], sources: dict[str, list[str]]) -> dict:
    markets = [config["experiment"]["market"] for config in configs]
    seed_map = {c["experiment"]["market"]: c["experiment"]["stage2_seeds"] for c in configs}
    model_map = {c["experiment"]["market"]: [c["experiment"].get("model", BASELINE_ID)] for c in configs}
    periods = {
        c["experiment"]["market"]: {
            name: [str(x) for x in c["data"][f"{name}_period"]]
            for name in ("train", "valid", "test")
        }
        for c in configs
    }
    enabled = all(len(seeds) >= 2 for seeds in seed_map.values())
    return {
        "schema_version": "1.0",
        "baseline": BASELINE_ID,
        "markets": markets,
        "models": model_map,
        "seeds": seed_map,
        "periods": periods,
        "prediction_sources": sources,
        "market_config": {c["experiment"]["market"]: market_metadata(c) for c in configs},
        "backtest": {
            "strategy": "Qlib TopkDropoutStrategy",
            **STRATEGY_KWARGS,
            "freq": "day",
            "account": ACCOUNT,
            "open_cost": EXCHANGE_KWARGS["open_cost"],
            "close_cost": EXCHANGE_KWARGS["close_cost"],
            "min_cost": EXCHANGE_KWARGS["min_cost"],
            "executor": "SimulatorExecutor",
            "return_field_is_gross": True,
            "return_semantics_evidence": (
                "Qlib 0.9.7 Account.update_portfolio_metrics sets return_rate="
                "(now_earning + now_cost) / last_account_value"
            ),
            "net_return_formula": "daily_ret_net = report['return'] - report['cost']",
        },
        "signal_alignment": {
            "signal_date": "t-1", "trade_date": "t", "qlib_internal_shift": 1,
            "adapter_manual_shift": False,
            "label_horizon": "Ref($close, -2) / Ref($close, -1) - 1; close(t+1) to close(t+2)",
        },
        "metric_convention": {
            "annualization": 252, "std_ddof": 1, "risk_free_rate": 0, "MAR_daily": 0,
            "ranking": "daily cross-sectional Pearson/Spearman; IR=mean/std(ddof=1), not annualized",
            "AR": "exp(mean(log1p(r_net))*252)-1",
            "STD": "std(log1p(r_net),ddof=1)*sqrt(252)",
            "MDD": "min([1,exp(cumsum(g))]/cummax-1)",
            "Sharpe": "sqrt(252)*mean(g)/std(g,ddof=1)",
            "Sortino": "sqrt(252)*mean(g)/sqrt(mean(min(g,0)^2)) over all days",
            "Calmar": "AR/abs(MDD)",
        },
        "ensemble": {
            "enabled": enabled, "methods": ["avg_none"] if enabled else [], "join": "inner",
            "normalize": "none", "score_formula": "arithmetic mean of aligned raw seed scores",
            "ranking_metrics_source": "recomputed ensemble score and aligned label",
        },
        "qlib_version": qlib.__version__,
        "data_version_or_cutoff": "Qlib provider data through at least 2025-12-31; no provider revision ID available",
        "git_commit": git_commit(),
    }


def write_manifest(out: Path, ensemble: bool) -> None:
    primary = {"seed_metrics": ["market", "model", "seed"], "aggregate_metrics": ["market", "model"]}
    files = {
        "seed_metrics": "metrics/seed_metrics.csv",
        "aggregate_metrics": "metrics/aggregate_metrics.csv",
        "seed_table": "tables/seed_mean_std.csv",
        "eval_config": "metadata/eval_config.json",
        "validation": "diagnostics/validation.json",
    }
    if ensemble:
        primary["ensemble_metrics"] = ["market", "model", "ensemble_method"]
        files.update({
            "ensemble_metrics": "metrics/ensemble_metrics.csv",
            "ensemble_table": "tables/ensemble.csv",
            "ensemble_curves": "curves/ensemble/*.csv",
        })
    manifest = {
        "schema_version": "1.0", "baseline": BASELINE_ID,
        "description": "FactorVQVAE standardized seed and raw-score ensemble evaluation",
        "primary_keys": primary, "files": files,
    }
    (out / "metadata" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=["configs/csi300.yaml", "configs/sp500.yaml"])
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    for directory in ("metrics", "tables", "curves/ensemble", "metadata", "diagnostics"):
        (out / directory).mkdir(parents=True, exist_ok=True)

    configs = [load_config(Path(path)) for path in args.configs]
    seed_rows, ensemble_rows = [], []
    sources: dict[str, list[str]] = {}
    for config in configs:
        market = config["experiment"]["market"]
        model = config["experiment"].get("model", BASELINE_ID)
        seeds = config["experiment"]["stage2_seeds"]
        data, evaluation = config["data"], config["evaluation"]
        start, end = map(str, data["test_period"])
        provider = str(Path(data["provider_uri"]).expanduser())
        init_market(provider, data["region"])
        calendar = expected_calendar(start, end)
        frames, paths = [], []
        for seed in seeds:
            path = prediction_path(config, seed)
            frame = load_prediction(path)
            frames.append(frame)
            relative = path.resolve().relative_to(root).as_posix()
            paths.append(relative)
            curve = run_standard_backtest(
                frame["score"], start_time=start, end_time=end,
                benchmark=evaluation["benchmark"], deal_price=evaluation["deal_price"],
                limit_threshold=evaluation["limit_threshold"],
            )
            if len(curve) != len(calendar):
                raise RuntimeError(f"{market} seed {seed}: curve/calendar length mismatch")
            seed_rows.append({
                "market": market, "model": model, "seed": seed,
                **ranking_metrics(frame), **portfolio_metrics(curve["daily_ret_net"]),
                "pred_path_or_ckpt_path": relative,
            })
        sources[market] = paths

        if len(seeds) >= 2:
            ensemble = ensemble_prediction(frames)
            curve = run_standard_backtest(
                ensemble["score"], start_time=start, end_time=end,
                benchmark=evaluation["benchmark"], deal_price=evaluation["deal_price"],
                limit_threshold=evaluation["limit_threshold"],
            )
            curve.to_csv(out / "curves" / "ensemble" / f"{market}_{model}.csv", index=False)
            ensemble_rows.append({
                "market": market, "model": model, "ensemble_method": "avg_none",
                **ranking_metrics(ensemble), **portfolio_metrics(curve["daily_ret_net"]),
                "seeds": ",".join(map(str, seeds)), "pred_paths": ",".join(paths),
            })

    seed_frame = pd.DataFrame(seed_rows, columns=SEED_COLUMNS).sort_values(["market", "model", "seed"])
    aggregate = aggregate_seed_metrics(seed_frame)
    seed_frame.to_csv(out / "metrics" / "seed_metrics.csv", index=False)
    aggregate.to_csv(out / "metrics" / "aggregate_metrics.csv", index=False)
    mean_std_table(aggregate).to_csv(out / "tables" / "seed_mean_std.csv", index=False)

    ensemble_enabled = bool(ensemble_rows)
    if ensemble_enabled:
        ensemble_frame = pd.DataFrame(ensemble_rows, columns=ENSEMBLE_COLUMNS).sort_values(
            ["market", "model", "ensemble_method"]
        )
        ensemble_frame.to_csv(out / "metrics" / "ensemble_metrics.csv", index=False)
        display = ensemble_frame.copy()
        for metric in RANKING_COLUMNS + PORTFOLIO_COLUMNS:
            display[metric] = display[metric].map(lambda value: f"{value:.4f}")
        display.to_csv(out / "tables" / "ensemble.csv", index=False)

    eval_config = build_eval_config(configs, sources)
    (out / "metadata" / "eval_config.json").write_text(json.dumps(eval_config, indent=2) + "\n")
    write_manifest(out, ensemble_enabled)
    print(f"generated={out}")


if __name__ == "__main__":
    main()
