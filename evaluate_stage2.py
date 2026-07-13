"""Evaluate an existing Stage2 checkpoint without retraining."""

from argparse import ArgumentParser
from pathlib import Path
import pickle

import torch
import yaml

from data.dataset import init_data_loader
from trainer.autoregressive import minGPT
from utils import run_inference


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config["train"]["seed"] = args.seed

    data = config["data"]
    test_path = (
        Path(data["data_path"])
        / data["prefix"]
        / f'{data["universe"]}_others_{data["window_size"]}_dl_test.pkl'
    )
    with test_path.open("rb") as f:
        test_data = pickle.load(f)
    test_data.config(fillna_type=data.get("window_fillna", "ffill"))
    test_loader = init_data_loader(
        test_data, shuffle=False, num_workers=config["train"]["num_workers"]
    )

    model = minGPT.load_from_checkpoint(
        args.checkpoint,
        config=config,
        n_train_samples=1,
        weights_only=False,
    ).eval()
    pred_df, _, _ = run_inference(model, test_loader)

    market = config["experiment"]["market"]
    vq_code = config["vqvae"]["num_factors"]
    hidden = config["transformer"]["hidden_size"]
    heads = config["transformer"]["heads"]
    layers = config["transformer"]["n_layers"]
    run_name = f"{market}_Stage2_VQ{vq_code}_Th{hidden}_h{heads}_l{layers}_sd{args.seed}"
    result_dir = Path(config["paths"]["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    output = result_dir / f"{run_name}.pkl"
    pred_df.to_pickle(output)
    print(f"saved={output}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
