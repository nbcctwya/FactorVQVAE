"""Build the cached Qlib datasets consumed by the two training stages.

The resulting tensor layout is [Alpha158, JKP13, label], i.e. 172 columns.
"""

from argparse import ArgumentParser
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import qlib
import yaml
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DataHandlerLP, TSDatasetH


JKP_COLUMNS = [
    "accruals",
    "debt_issuance",
    "investment",
    "low_leverage",
    "low_risk",
    "momentum",
    "profit_growth",
    "profitability",
    "quality",
    "seasonality",
    "short_term_reversal",
    "size",
    "value",
]

def load_jkp(path: Path, train_period) -> pd.DataFrame:
    raw = pd.read_csv(path, parse_dates=["date"])
    market = raw.pivot(index="date", columns="name", values="ret").sort_index()
    missing = set(JKP_COLUMNS) - set(market.columns)
    if missing:
        raise ValueError(f"JKP file is missing factors: {sorted(missing)}")
    market = market[JKP_COLUMNS]

    # Match Qlib's RobustZScoreNorm using training-period statistics only.
    fit = market.loc[train_period[0] : train_period[1]]
    center = fit.median(axis=0)
    scale = (fit - center).abs().median(axis=0) * 1.4826 + 1e-12
    return ((market - center) / scale).clip(-3, 3).astype("float32")


def attach_market(df: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Expected Qlib feature/label MultiIndex columns")

    dates = pd.DatetimeIndex(df.index.get_level_values("datetime"))
    aligned = market.reindex(dates)
    if aligned.isna().any().any():
        missing_dates = pd.DatetimeIndex(dates[aligned.isna().any(axis=1)]).unique()
        raise ValueError(f"JKP data missing for Qlib dates: {missing_dates[:10].tolist()}")

    prior = pd.DataFrame(aligned.to_numpy(), index=df.index)
    prior.columns = pd.MultiIndex.from_product([["prior"], JKP_COLUMNS])
    feature = df.loc[:, "feature"]
    label = df.loc[:, "label"]
    feature.columns = pd.MultiIndex.from_product([["feature"], feature.columns])
    label.columns = pd.MultiIndex.from_product([["label"], label.columns])
    return pd.concat([feature, prior, label], axis=1).astype("float32")


def purge_segment_tail(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """Remove prediction dates whose label realization crosses a split boundary."""
    if days <= 0 or df.empty:
        return df
    dates = pd.DatetimeIndex(df.index.get_level_values("datetime")).unique().sort_values()
    if len(dates) <= days:
        raise ValueError(f"segment has only {len(dates)} dates; cannot purge {days}")
    return df.loc[df.index.get_level_values("datetime") < dates[-days]]


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--segments", nargs="+", choices=["train", "valid", "test"],
        default=["train", "valid", "test"],
        help="Only rebuild the selected cached segments.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    data = config["data"]
    segments = {
        name: tuple(str(value) for value in data[f"{name}_period"])
        for name in ("train", "valid", "test")
    }
    provider_uri = str(Path(data["provider_uri"]).expanduser())
    output_dir = Path(data["data_path"]) / data["prefix"]
    window_fillna = data.get("window_fillna", "ffill")
    if window_fillna not in ("none", "ffill"):
        raise ValueError("data.window_fillna must be 'none' or leakage-safe 'ffill'")

    qlib.init(provider_uri=provider_uri, region=data["region"])
    handler = Alpha158(
        instruments=data["universe"],
        start_time=str(data["handler_start"]),
        end_time=str(data["handler_end"]),
        fit_start_time=segments["train"][0],
        fit_end_time=segments["train"][1],
        infer_processors=[
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ],
        learn_processors=[
            {"class": "DropnaLabel"},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
        ],
        label=[data["label"]],
    )
    market = load_jkp(Path(data["jkp_path"]), segments["train"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in args.segments:
        period = segments[name]
        # Every model_label must have the same CSRankNorm transformation used
        # to train Stage1.  Test additionally keeps the raw return strictly for
        # metrics/backtesting; it is never passed to the encoder.
        data_key = DataHandlerLP.DK_L
        frame = handler.fetch(
            slice(*period), col_set=["feature", "label"], data_key=data_key
        )
        # Labels near a train/validation boundary use prices from the following
        # split.  Keep the requested calendar split, but embargo those samples.
        if name in ("train", "valid"):
            frame = purge_segment_tail(frame, int(data.get("purge_label_days", 0)))
        frame = attach_market(frame, market)
        if name == "test":
            raw = handler.fetch(
                slice(*period), col_set=["label"], data_key=DataHandlerLP.DK_I
            ).reindex(frame.index)
            raw_label = raw.loc[:, "label"]
            raw_label.columns = pd.MultiIndex.from_product(
                [["raw_label"], raw_label.columns]
            )
            frame = pd.concat([frame, raw_label.astype("float32")], axis=1)
        static_handler = DataHandlerLP.from_df(frame)
        dataset = TSDatasetH(
            handler=static_handler,
            segments={name: period},
            step_len=data["window_size"],
        )
        sampler = dataset.prepare(name, data_key=DataHandlerLP.DK_I)
        sampler.config(fillna_type=window_fillna)

        path = output_dir / f"{data['universe']}_others_{data['window_size']}_dl_{name}.pkl"
        with path.open("wb") as f:
            pickle.dump(sampler, f, protocol=pickle.HIGHEST_PROTOCOL)
        sample = sampler[0]
        expected_width = 173 if name == "test" else 172
        if sample.shape[-1] != expected_width:
            raise RuntimeError(f"Unexpected {name} tensor shape: {sample.shape}")
        print(f"{name}: {len(sampler)} samples, sample={sample.shape}, saved={path}")


if __name__ == "__main__":
    main()
