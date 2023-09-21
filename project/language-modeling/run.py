import sys
import yaml
import math
import pickle
import logging
import numpy as np
import pandas as pd
from os import listdir, makedirs
from os.path import join, isfile, isdir
from argparse import ArgumentParser


import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from cli import launch_tui, get_parser, train, evaluate, inference, load_config

from typing import Any

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    config = load_config(args.config_path)

    if args.train:
        train(config)
    if args.evaluate:
        metrics = evaluate(config, args.dump_outputs)
        print(metrics)
    if args.inference:
        inference_config = load_config(args.inference_config_path)
        if not args.interactive:
            generated = inference(config, inference_config, args.prompt)
            print(generated)
        else:
            launch_tui(config, inference_config)
    if not any([args.train, args.evaluate, args.inference]):
        raise ValueError("Please provide a supported mode flag ('-t', '-e', '-i')")
    