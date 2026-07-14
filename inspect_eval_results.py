"""Validate Baseline Results Protocol v1.0 artifacts; exits nonzero on failure."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

from baseline_results.metrics import PORTFOLIO_COLUMNS, RANKING_COLUMNS, ranking_metrics


class Validator:
    def __init__(self, root: Path):
        self.root = root
        self.checks: list[dict[str, object]] = []

    def check(self, name: str, condition: bool, detail: str) -> None:
        self.checks.append({"name": name, "passed": bool(condition), "detail": detail})

    def attempt(self, name: str, function) -> None:
        try:
            detail = function()
            self.check(name, True, str(detail or "ok"))
        except Exception as exc:  # validator must report every independent failure
            self.check(name, False, f"{type(exc).__name__}: {exc}")

    def write(self) -> bool:
        passed = sum(bool(item["passed"]) for item in self.checks)
        report = {
            "passed": passed == len(self.checks),
            "passes": passed,
            "failures": len(self.checks) - passed,
            "checks": self.checks,
        }
        target = self.root / "diagnostics" / "validation.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2) + "\n")
        return bool(report["passed"])


def assert_close(actual, expected, *, atol=1e-12, rtol=1e-10, context="values") -> None:
    if not np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True):
        delta = np.nanmax(np.abs(np.asarray(actual, float) - np.asarray(expected, float)))
        raise AssertionError(f"{context} differ; max_abs_delta={delta}")


def independent_portfolio_metrics(returns: pd.Series) -> dict[str, float | int]:
    r = np.asarray(returns, dtype=float)
    if not np.isfinite(r).all() or (r <= -1).any() or not len(r):
        raise ValueError("net returns must be finite, nonempty, and greater than -1")
    g = np.log1p(r)
    mean, std = g.mean(), g.std(ddof=1)
    nav_with_origin = np.r_[1.0, np.exp(np.cumsum(g))]
    downside = np.sqrt(np.mean(np.minimum(g, 0) ** 2))
    ar = np.exp(mean * 252) - 1
    mdd = np.min(nav_with_origin / np.maximum.accumulate(nav_with_origin) - 1)
    return {
        "AR": ar, "STD": std * np.sqrt(252), "MDD": mdd,
        "Sharpe": np.sqrt(252) * mean / std,
        "Sortino": np.sqrt(252) * mean / downside,
        "Calmar": ar / abs(mdd), "num_test_days": len(r),
    }


def parse_mean_std(value: str) -> tuple[float, float]:
    left, right = value.split("±")
    return float(left.strip()), float(right.strip())


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    root = Path(args.results).resolve()
    validator = Validator(root)
    diagnostic = root / "diagnostics" / "validation.json"
    diagnostic.parent.mkdir(parents=True, exist_ok=True)
    diagnostic.write_text('{"passed": false, "passes": 0, "failures": 1, "checks": []}\n')

    try:
        manifest = json.loads((root / "metadata" / "manifest.json").read_text())
        config = json.loads((root / "metadata" / "eval_config.json").read_text())
    except Exception as exc:
        validator.check("metadata_load", False, str(exc))
        raise SystemExit(0 if validator.write() else 1)

    def files_exist():
        missing = []
        for value in manifest["files"].values():
            if "*" in value:
                if not list(root.glob(value)):
                    missing.append(value)
            elif not (root / value).is_file():
                missing.append(value)
        if missing:
            raise AssertionError(f"missing manifest files: {missing}")
        return f"{len(manifest['files'])} manifest entries resolve"
    validator.attempt("manifest_files_exist", files_exist)

    try:
        seed = pd.read_csv(root / manifest["files"]["seed_metrics"])
        aggregate = pd.read_csv(root / manifest["files"]["aggregate_metrics"])
        seed_table = pd.read_csv(root / manifest["files"]["seed_table"])
    except Exception as exc:
        validator.check("core_csv_load", False, str(exc))
        raise SystemExit(0 if validator.write() else 1)

    expected_seed_columns = [
        "market", "model", "seed", *RANKING_COLUMNS, *PORTFOLIO_COLUMNS,
        "num_test_days", "pred_path_or_ckpt_path",
    ]
    validator.check("seed_column_order", list(seed.columns) == expected_seed_columns, str(list(seed.columns)))

    expected_keys = {
        (market, model, int(seed_value))
        for market, seeds in config["seeds"].items()
        for model in config["models"][market]
        for seed_value in seeds
    }
    actual_keys = set(seed[["market", "model", "seed"]].itertuples(index=False, name=None))
    validator.check("seed_completeness", actual_keys == expected_keys, f"expected={len(expected_keys)}, actual={len(actual_keys)}")
    validator.check("seed_primary_key_unique", not seed.duplicated(["market", "model", "seed"]).any(), f"rows={len(seed)}")
    numeric = seed[[*RANKING_COLUMNS, *PORTFOLIO_COLUMNS, "num_test_days"]].to_numpy(float)
    validator.check("seed_metrics_finite", np.isfinite(numeric).all(), "all seed numeric metrics must be finite")
    validator.check("seed_metric_bounds", bool(
        seed["IC"].abs().le(1).all() and seed["RankIC"].abs().le(1).all()
        and seed["STD"].ge(0).all() and seed["MDD"].le(0).all()
    ), "|IC|,|RankIC|<=1; STD>=0; MDD<=0")

    def aggregate_exact():
        for (market, model), group in seed.groupby(["market", "model"]):
            row = aggregate[(aggregate.market == market) & (aggregate.model == model)]
            if len(row) != 1:
                raise AssertionError(f"aggregate row count for {market}/{model}: {len(row)}")
            for metric in RANKING_COLUMNS + PORTFOLIO_COLUMNS:
                assert_close(row[f"{metric}_mean"].iloc[0], group[metric].mean(), context=f"{metric}_mean")
                assert_close(row[f"{metric}_std"].iloc[0], group[metric].std(ddof=1), context=f"{metric}_std")
        return f"verified {len(aggregate)} groups"
    validator.attempt("aggregate_recomputed", aggregate_exact)

    def seed_table_matches():
        for _, row in aggregate.iterrows():
            shown = seed_table[(seed_table.market == row.market) & (seed_table.model == row.model)]
            if len(shown) != 1:
                raise AssertionError("seed display group missing or duplicated")
            for metric in RANKING_COLUMNS + PORTFOLIO_COLUMNS:
                mean, std = parse_mean_std(shown[metric].iloc[0])
                assert_close(mean, row[f"{metric}_mean"], atol=5.0001e-5, rtol=0, context=f"table {metric} mean")
                assert_close(std, row[f"{metric}_std"], atol=5.0001e-5, rtol=0, context=f"table {metric} std")
        return "four-decimal seed table matches numeric aggregate"
    validator.attempt("seed_table_matches", seed_table_matches)

    fixed = {
        "strategy": "Qlib TopkDropoutStrategy", "topk": 30, "n_drop": 5,
        "method_sell": "bottom", "method_buy": "top", "hold_thresh": 1,
        "only_tradable": False, "forbid_all_trade_at_limit": True, "risk_degree": 0.95,
        "account": 100000000, "open_cost": 0.0005, "close_cost": 0.0015,
        "min_cost": 0.0, "freq": "day",
    }
    validator.check("fixed_backtest_config", all(config["backtest"].get(k) == v for k, v in fixed.items()), str(fixed))
    alignment = config.get("signal_alignment", {})
    validator.check("signal_alignment", alignment.get("signal_date") == "t-1" and alignment.get("trade_date") == "t" and alignment.get("qlib_internal_shift") == 1 and alignment.get("adapter_manual_shift") is False, str(alignment))

    def prediction_coverage_and_paths():
        from baseline_results.qlib_adapter import expected_calendar, init_market

        project = root.parent
        for market, group in seed.groupby("market"):
            market_cfg = config["market_config"][market]
            init_market(str(Path(market_cfg["provider_uri"]).expanduser()), market_cfg["region"])
            start, end = config["periods"][market]["test"]
            calendar = expected_calendar(start, end)
            for path_text in group["pred_path_or_ckpt_path"]:
                relative = Path(path_text)
                path = project / relative
                if relative.is_absolute() or not path.is_file():
                    raise AssertionError(f"nonportable or missing prediction path: {path_text}")
                pred = pd.read_pickle(path)
                dates = pd.DatetimeIndex(pred.index.get_level_values("datetime").unique()).sort_values()
                if not dates.equals(calendar):
                    raise AssertionError(f"{path_text}: prediction dates do not equal test calendar")
            if not group["num_test_days"].eq(len(calendar)).all():
                raise AssertionError(f"{market}: num_test_days does not equal calendar")
        return "all prediction paths portable and cover their complete test calendars"
    validator.attempt("prediction_test_coverage", prediction_coverage_and_paths)

    if config["ensemble"]["enabled"]:
        ensemble = pd.read_csv(root / manifest["files"]["ensemble_metrics"], dtype={"seeds": str, "pred_paths": str})
        ensemble_table = pd.read_csv(root / manifest["files"]["ensemble_table"], dtype={"seeds": str, "pred_paths": str})
        expected_ensemble = {
            (market, model, method)
            for market in config["markets"] for model in config["models"][market]
            for method in config["ensemble"]["methods"]
        }
        actual_ensemble = set(ensemble[["market", "model", "ensemble_method"]].itertuples(index=False, name=None))
        validator.check("ensemble_completeness", actual_ensemble == expected_ensemble and not ensemble.duplicated(["market", "model", "ensemble_method"]).any(), f"expected={len(expected_ensemble)}, actual={len(actual_ensemble)}")
        validator.check("ensemble_metrics_finite", np.isfinite(ensemble[[*RANKING_COLUMNS, *PORTFOLIO_COLUMNS, "num_test_days"]].to_numpy(float)).all(), "all ensemble numeric metrics finite")

        def ensemble_table_matches():
            for metric in RANKING_COLUMNS + PORTFOLIO_COLUMNS:
                assert_close(ensemble_table[metric], ensemble[metric], atol=5.0001e-5, rtol=0, context=f"ensemble table {metric}")
            return "four-decimal ensemble table matches numeric metrics"
        validator.attempt("ensemble_table_matches", ensemble_table_matches)

        def validate_curves():
            for _, row in ensemble.iterrows():
                path = root / "curves" / "ensemble" / f"{row.market}_{row.model}.csv"
                curve = pd.read_csv(path, parse_dates=["datetime"])
                expected_columns = ["datetime", "daily_ret_gross", "cost", "daily_ret_net", "bench_ret", "nav", "bench_nav"]
                if list(curve.columns) != expected_columns or curve.datetime.duplicated().any() or not curve.datetime.is_monotonic_increasing:
                    raise AssertionError(f"invalid curve schema/order: {path}")
                if not np.isfinite(curve.iloc[:, 1:].to_numpy(float)).all():
                    raise AssertionError(f"NaN/Inf curve: {path}")
                assert_close(curve.daily_ret_net, curve.daily_ret_gross - curve.cost, context="net=gross-cost")
                assert_close(curve.nav, (1 + curve.daily_ret_net).cumprod(), context="nav")
                assert_close(curve.bench_nav, (1 + curve.bench_ret).cumprod(), context="bench_nav")
                recomputed = independent_portfolio_metrics(curve.daily_ret_net)
                for metric in PORTFOLIO_COLUMNS + ("num_test_days",):
                    assert_close(row[metric], recomputed[metric], context=f"ensemble {metric}")
            return f"verified {len(ensemble)} curves and all portfolio formulas"
        validator.attempt("ensemble_curves_and_metrics", validate_curves)

        def ensemble_ranking_source():
            project = root.parent
            for _, row in ensemble.iterrows():
                paths = row.pred_paths.split(",")
                frames = [pd.read_pickle(project / path) for path in paths]
                joined = pd.concat(
                    [frame.rename(columns={"score": f"score_{i}", "label": f"label_{i}"}) for i, frame in enumerate(frames)],
                    axis=1, join="inner",
                )
                score = joined.filter(regex=r"^score_").mean(axis=1)
                label = joined.filter(regex=r"^label_").iloc[:, 0]
                recomputed = ranking_metrics(pd.DataFrame({"score": score, "label": label}))
                for metric in RANKING_COLUMNS:
                    assert_close(row[metric], recomputed[metric], context=f"ensemble {metric}")
            return "ranking metrics recomputed from aligned raw seed scores and labels"
        validator.attempt("ensemble_ranking_recomputed", ensemble_ranking_source)

    ok = validator.write()
    print(f"passed={ok} checks={len(validator.checks)} failures={sum(not c['passed'] for c in validator.checks)}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
