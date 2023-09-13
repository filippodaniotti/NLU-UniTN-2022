import yaml
import math
import pickle
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
            tbptt_config = config["experiment"].get("tbptt_config", None),
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
        batch_size = batch_size,
    )

def get_logger(config: dict[str, Any]) -> pl.loggers.TensorBoardLogger:
    return pl.loggers.TensorBoardLogger(
        config["results"]["logs_path"],
        config["experiment"]["experiment_name"]
    )
    
def get_cost_function(config: dict[str, Any]) -> nn.Module:
    return nn.CrossEntropyLoss(ignore_index=config["dataset"]["pad_value"])



def train(config: dict[str, Any]):
    seed_everything(config["experiment"]["seed"])
    ptb = get_data_module(config)
    ptb.prepare_data() 
    logger = get_logger(config)
    trainer = pl.Trainer(max_epochs=config["experiment"]["epochs"], logger=logger)
    model = get_model(config, ptb.vocab_size, train=True)
    trainer.fit(model=model, datamodule=ptb)

def evaluate(config: dict[str, Any], dump_outputs: bool | None):
    ptb = get_data_module(config, batch_size=1)
    ptb.prepare_data()
    trainer = pl.Trainer(logger=False)
    model = get_model(config, ptb.vocab_size, train=False)

    def _run_loop(split: str, split_fn: callable):
        ptb.setup(split)
        split_fn(model=model, datamodule=ptb, verbose=False)
        loss_mean = float(pd.DataFrame(model.outputs)["loss"].mean())
        split = split[0].upper() + split[1:]
        metrics[split] = [loss_mean, math.exp(loss_mean)]

    metrics = {}
    _run_loop("valid", trainer.validate)
    _run_loop("test", trainer.test)

    if dump_outputs:
        outputs_path = join(config["results"]["results_path"], "outputs")
        if not isdir(config["results"]["results_path"]) or not isdir(outputs_path):
            makedirs(outputs_path)
        results_path = join(outputs_path, f'{config["experiment"]["experiment_name"]}.pkl')
        SequenceModelWrapper.dump_outputs(model.outputs, results_path)

    return pd.DataFrame(metrics, index=["Loss", "Perplexity"])

def inference(config: dict[str, Any], inf_config: dict[str, Any], prompt: str):
    lang = load_lang(join(*inf_config["lang_path"]))
    model = get_model(config, len(lang), train=False)

    temperatures = [0.5, 0.7, 0.75, 0.8, 1.0]
    for temp in temperatures:
        generated = model.generate(
            prompt, 
            lang, 
            mode=inf_config["mode"], 
            max_len=inf_config["max_length"],
            allow_unk=inf_config["allow_unk"],
            temperature=temp,
            device=get_device())
        print(f"t:{temp} => {generated}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Base interface for training, evaluation and inference")
    parser.add_argument(
        "-c", 
        "--config", 
        type=str, 
        dest="config_path", 
        help="Path of configuration file"
    )
    parser.add_argument(
        "-t", 
        "--t", 
        action="store_true", 
        dest="train", 
        help="Flag for train mode"
    )
    parser.add_argument(
        "-e", 
        "--evaluate", 
        action="store_true", 
        dest="evaluate", 
        help="Flag for evaluation mode"
    )
    parser.add_argument(
        "-d", 
        "--dump-results", 
        action="store_true", 
        dest="dump_outputs", 
        help="Flag for dumping test outputs object after test run"
    )
    parser.add_argument(
        "-i", 
        "--inference", 
        action="store_true", 
        dest="inference", 
        help="Flag for inference mode"
    )
    parser.add_argument(
        "-ic", 
        "--inference-config", 
        type=str, 
        dest="inference_config_path", 
        default=join("configs", "inference.yaml"),
        help="Path of inference configuration file. Defaults to 'configs/inference.yaml''"
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        dest="prompt",
        default="the",
        help="Prompt for inference mode",
    )

    args = parser.parse_args()
    config = load_config(args.config_path)
    if args.train:
        train(config)
    if args.evaluate:
        metrics = evaluate(config, args.dump_outputs)
        print(metrics)
    if args.inference:
        inference_config = load_config(args.inference_config_path)
        inference(config, inference_config, args.prompt)
    if not any([args.train, args.evaluate, args.inference]):
        raise ValueError("Please provide a supported mode flag ('-t', '-e', '-i')")
    