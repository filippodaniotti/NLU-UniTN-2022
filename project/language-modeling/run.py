import yaml
from argparse import ArgumentParser

import torch.nn as nn
import pytorch_lightning as pl

from data.data_module import PennTreebank 
from models.lstm import BaselineLSTM
from models.wrapper import SequenceModelWrapper

from typing import Any

def get_model(config: dict[str, Any], vocab_size: int) -> nn.Module:
    if config["experiment"]["model"] == "baseline":
        return BaselineLSTM(
            num_classes = vocab_size,
            embedding_dim = config["model"]["embedding_dim"],
            hidden_dim = config["model"]["hidden_dim"],
            num_layers = config["model"]["num_layers"],
            p_dropout = config["model"]["p_dropout"]
        )
    else:
        raise ValueError("Provided model not available.")

def get_cost_function(config: dict[str, Any]) -> nn.Module:
    return nn.CrossEntropyLoss(ignore_index=config["dataset"]["pad_value"])

def train(config: dict[str, Any]):
    ptb = PennTreebank(
        download_url = config["dataset"]["ds_url"], 
        data_dir = config["dataset"]["ds_path"], 
        batch_size = config["experiment"]["batch_size"]
    )
    ptb.prepare_data()
    logger = pl.loggers.TensorBoardLogger(
        config["results"]["logs_path"], 
        config["experiment"]["experiment_name"]
    )
    trainer = pl.Trainer(max_epochs=config["experiment"]["epochs"], logger=logger)
    model = SequenceModelWrapper(
        model = get_model(config, ptb.vocab_size),
        cost_function = get_cost_function(config),
        optimizer = config["experiment"]["optimizer"],
        learning_rate = float(config["experiment"]["learning_rate"])
    )
    trainer.fit(model=model, datamodule=ptb)

if __name__ == "__main__":
    parser = ArgumentParser(description="Base interface")
    parser.add_argument(
        "-t", "--train", action="store_true", help="Train model flag"
    )
    parser.add_argument(
        "-c", "--config", type=str, dest="config_path", help="Path of configuration file"
    )

    args = parser.parse_args()
    with open(args.config_path) as config_file:
        config = yaml.safe_load(config_file)
    if args.train:
        train(config)