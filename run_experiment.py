"""Configuration-driven FactorVQVAE experiment runner."""

from argparse import ArgumentParser
from pathlib import Path
import subprocess
import sys

import yaml


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["prepare", "stage1", "stage2"],
        default=["prepare", "stage1", "stage2"],
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open() as f:
        config = yaml.safe_load(f)
    declared_seeds = config["experiment"]["stage2_seeds"]
    seeds = args.seeds if args.seeds is not None else declared_seeds
    invalid = sorted(set(seeds) - set(declared_seeds))
    if invalid:
        raise ValueError(f"Stage2 seeds not declared in config: {invalid}")

    python = sys.executable
    if "prepare" in args.stages:
        run(python, "prepare_data.py", "--config", str(config_path))
    if "stage1" in args.stages:
        run(python, "stage1.py", "--config", str(config_path))
    if "stage2" in args.stages:
        checkpoint = Path(config["paths"]["checkpoint_dir"]) / config["transformer"]["saved_model"]
        if not checkpoint.exists():
            raise FileNotFoundError(f"Stage1 checkpoint does not exist: {checkpoint}")
        for seed in seeds:
            run(
                python,
                "stage2_gpt.py",
                "--config",
                str(config_path),
                "--seed",
                str(seed),
            )


if __name__ == "__main__":
    main()
