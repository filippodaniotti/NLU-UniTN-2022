import yaml
import pickle
import numpy as np
from os import listdir
from os.path import join, isfile

import torch
import torch.nn as nn
import pytorch_lightning as pl

from data import PennTreebank, Lang
from models import BaselineLSTM, MerityLSTM, MogrifierLSTM, SequenceModelWrapper

from typing import Any

def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path) as config_file:
        return yaml.safe_load(config_file)

def load_lang(lang_path: str) -> Lang:
    with open(lang_path, "rb") as lang_file:
        return pickle.load(lang_file)

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed)

def get_weights_path(config: dict[str, Any]) -> str:
    if config["experiment"].get("checkpoint_path", None):
        return join(*config["experiment"]["checkpoint_path"])
    else:
        paths = []
        # check on results_path
        paths.append(join(config["results"]["results_path"], "weights", f'{config["experiment"]["experiment_name"]}.ckpt'))
        # check on logs_path
        tmp = join(config["results"]["logs_path"], config["experiment"]["experiment_name"], "version_0", "checkpoints")
        tmp_fn = [f for f in listdir(tmp) if f.endswith(".ckpt")][0]
        paths.append(join(tmp, tmp_fn))
        for path in paths:
            if isfile(path):
                return path
        else:
            raise ValueError("No checkpoint found.")
    

def get_model_core(config: dict[str, Any], vocab_size: int) -> nn.Module:
    if config["experiment"]["model"] == "baseline":
        return BaselineLSTM(
            num_classes = vocab_size,
            embedding_dim = config["model"]["embedding_dim"],
            hidden_dim = config["model"]["hidden_dim"],
            num_layers = config["model"]["num_layers"],
            p_dropout = config["model"]["p_dropout"],
            pad_value = config["dataset"]["pad_value"]
        )
    elif config["experiment"]["model"] == "merity":
        return MerityLSTM(
            num_classes = vocab_size,
            embedding_dim = config["model"]["embedding_dim"],
            hidden_dim = config["model"]["hidden_dim"],
            num_layers = config["model"]["num_layers"],
            locked_dropout= bool(config["model"]["locked_dropout"]),
            p_lockdrop = config["model"]["p_lockdrop"],
            embedding_dropout = bool(config["model"]["embedding_dropout"]),
            p_embdrop = config["model"]["p_embdrop"],
            weight_dropout = bool(config["model"]["weight_dropout"]),
            p_lstmdrop = config["model"]["p_lstmdrop"],
            p_hiddrop = config["model"]["p_hiddrop"],
            init_weights = bool(config["model"]["init_weights"]),
            tie_weights = bool(config["model"]["tie_weights"]),
            pad_value = config["dataset"]["pad_value"]
        )
    elif config["experiment"]["model"] == "mogrifier":
        return MogrifierLSTM(
            num_classes = vocab_size,
            embedding_dim = config["model"]["embedding_dim"],
            hidden_dim = config["model"]["hidden_dim"],
            num_layers = config["model"]["num_layers"],
            mogrify_steps = config["model"]["mogrify_steps"],
            tie_weights = bool(config["model"]["tie_weights"]),
            p_dropout = config["model"]["p_dropout"],
            pad_value = config["dataset"]["pad_value"]
        )
    else:
        raise ValueError(f"Provided model '{config['experiment']['model']}' not available.")

def get_model(
        config: dict[str, Any], 
        vocab_size: int,
        train: bool = True,) -> SequenceModelWrapper:
    batch_size = config["experiment"]["batch_size"] if train else 1
    if train:
        return SequenceModelWrapper(
            model = get_model_core(config, vocab_size),
            cost_function = get_cost_function(config),
            optimizer = config["experiment"]["optimizer"],
            learning_rate = float(config["experiment"]["learning_rate"]),
            ntasgd = config["experiment"].get("ntasgd", -1),
            asgd_lr = float(config["experiment"].get("asgd_lr", .0)),
            tbptt = bool(config["experiment"].get("tbptt", False)),
            batch_size = batch_size,
            evaluate = not train,
        )
    else:
        return SequenceModelWrapper.load_model(
            checkpoint_path = get_weights_path(config),
            map_location = get_device(),
            model = get_model_core(config, vocab_size),
            cost_function = get_cost_function(config),
        )

def get_data_module(
        config: dict[str, Any], 
        batch_size: int | None = None) -> PennTreebank:
    batch_size = batch_size or config["experiment"]["batch_size"]
    return PennTreebank(
        download_url = config["dataset"]["ds_url"],
        data_dir = config["dataset"]["ds_path"],
        pad_value = config["dataset"]["pad_value"],
        tbptt = bool(config["experiment"].get("tbptt", False)),
        tbptt_config = config["experiment"].get("tbptt_config", None),
        batch_size = batch_size,
    )

def get_logger(config: dict[str, Any]) -> pl.loggers.TensorBoardLogger:
    return pl.loggers.TensorBoardLogger(
        config["results"]["logs_path"],
        config["experiment"]["experiment_name"]
    )
    
def get_cost_function(config: dict[str, Any]) -> nn.Module:
    return nn.CrossEntropyLoss(ignore_index=config["dataset"]["pad_value"])

