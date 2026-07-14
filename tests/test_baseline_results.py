from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from baseline_results.metrics import portfolio_metrics
from baseline_results.qlib_adapter import ACCOUNT, EXCHANGE_KWARGS, STRATEGY_KWARGS, run_standard_backtest


class PortfolioMetricTests(unittest.TestCase):
    def test_first_day_loss_is_drawdown(self):
        metrics = portfolio_metrics(pd.Series([-0.10, 0.02]))
        self.assertAlmostEqual(metrics["MDD"], -0.10)

    def test_identical_negative_days_have_defined_sortino(self):
        metrics = portfolio_metrics(pd.Series([-0.01, -0.01]))
        self.assertTrue(np.isfinite(metrics["Sortino"]))
        self.assertAlmostEqual(metrics["Sortino"], -math.sqrt(252))

    def test_total_loss_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "greater than -1"):
            portfolio_metrics(pd.Series([0.01, -1.0]))

    def test_independent_manual_vector(self):
        returns = np.array([0.01, -0.02, 0.005, -0.01, 0.015])
        actual = portfolio_metrics(returns)
        g = np.log(1 + returns)
        nav = np.r_[1.0, np.exp(np.cumsum(g))]
        expected = {
            "AR": np.exp(g.mean() * 252) - 1,
            "STD": g.std(ddof=1) * np.sqrt(252),
            "MDD": np.min(nav / np.maximum.accumulate(nav) - 1),
            "Sharpe": np.sqrt(252) * g.mean() / g.std(ddof=1),
            "Sortino": np.sqrt(252) * g.mean() / np.sqrt(np.mean(np.minimum(g, 0) ** 2)),
        }
        expected["Calmar"] = expected["AR"] / abs(expected["MDD"])
        for name, value in expected.items():
            self.assertAlmostEqual(actual[name], value, places=12)


class BacktestConfigurationTests(unittest.TestCase):
    @patch("baseline_results.qlib_adapter.validate_signal_coverage")
    @patch("baseline_results.qlib_adapter.backtest")
    def test_seed_and_ensemble_use_complete_fixed_configuration(self, mocked_backtest, mocked_coverage):
        dates = pd.date_range("2023-01-03", periods=2)
        report = pd.DataFrame(
            {"return": [0.01, -0.01], "cost": [0.001, 0.0], "bench": [0.002, -0.001]},
            index=dates,
        )
        mocked_backtest.return_value = ({"1day": (report, {})}, {})
        index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["datetime", "instrument"])
        seed_signal = pd.Series([1.0, 0.0, 0.8, 0.2], index=index)
        ensemble_signal = pd.Series([0.9, 0.1, 0.7, 0.3], index=index)

        for signal in (seed_signal, ensemble_signal):
            curve = run_standard_backtest(
                signal, start_time="2023-01-01", end_time="2025-12-31",
                benchmark="SH000300", deal_price="close", limit_threshold=0.095,
            )
            self.assertAlmostEqual(curve.loc[0, "daily_ret_net"], 0.009)

        self.assertEqual(mocked_backtest.call_count, 2)
        for call in mocked_backtest.call_args_list:
            kwargs = call.kwargs
            self.assertEqual(kwargs["start_time"], "2023-01-01")
            self.assertEqual(kwargs["end_time"], "2025-12-31")
            self.assertEqual(kwargs["account"], 100_000_000)
            self.assertEqual(kwargs["strategy"]["kwargs"] | {"signal": None}, STRATEGY_KWARGS | {"signal": None})
            self.assertEqual(kwargs["executor"]["kwargs"]["time_per_step"], "day")
            for key, value in EXCHANGE_KWARGS.items():
                self.assertEqual(kwargs["exchange_kwargs"][key], value)


if __name__ == "__main__":
    unittest.main()
