# yaml
from argparse import ArgumentParser
import os
from pathlib import Path
import yaml

def get_root_dir():
    return Path(__file__).parent.parent

def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help="Path to a market experiment configuration file.")
    parser.add_argument('--seed', type=int, default=None,
                        help="Optional Stage2 seed selected from experiment.stage2_seeds")
    return parser.parse_args()


def load_yaml_param_settings(yaml_fname: str):
    """
    :param yaml_fname: .yaml file that consists of hyper-parameter settings.
    """
    stream = open(yaml_fname, 'r')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    return config
