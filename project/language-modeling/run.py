import yaml
import math
import pickle
import numpy as np
import pandas as pd
from os.path import join
from argparse import ArgumentParser


import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from data.data_module import PennTreebank 
from models.lstm import BaselineLSTM
from models.merity import MerityLSTM
from models.wrapper import SequenceModelWrapper

from typing import Any

def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path) as config_file:
        return yaml.safe_load(config_file)

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed)

def get_model(config: dict[str, Any], vocab_size: int) -> nn.Module:
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
    else:
        raise ValueError("Provided model not available.")
    

def get_cost_function(config: dict[str, Any]) -> nn.Module:
    return nn.CrossEntropyLoss(ignore_index=config["dataset"]["pad_value"])

def train(config: dict[str, Any]):
    seed_everything(config["experiment"]["seed"])
    ptb = PennTreebank(
        download_url = config["dataset"]["ds_url"], 
        data_dir = config["dataset"]["ds_path"], 
        batch_size = config["experiment"]["batch_size"],
        tbptt= bool(config["experiment"]["tbptt"]),
        tbptt_config = config["experiment"]["tbptt_config"],
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
        learning_rate = float(config["experiment"]["learning_rate"]),
        ntasgd = config["experiment"]["ntasgd"],
        asgd_lr = float(config["experiment"]["asgd_lr"]),
        tbptt = bool(config["experiment"]["tbptt"]),
        tbptt_config = config["experiment"]["tbptt_config"],
        batch_size = config["experiment"]["batch_size"],
    )
    trainer.fit(model=model, datamodule=ptb)

def evaluate(config: dict[str, Any], dump_results: bool | None):
    ptb = PennTreebank(
        download_url = config["dataset"]["ds_url"], 
        data_dir = config["dataset"]["ds_path"], 
        batch_size = 1,
        tbptt= bool(config["experiment"]["tbptt"]),
    )
    ptb.prepare_data()
    trainer = pl.Trainer(logger=False)
    model = SequenceModelWrapper.load_model(
        checkpoint_path = join(*config["experiment"]["checkpoint_path"]),
        map_location = get_device(),
        model = get_model(config, ptb.vocab_size),
        cost_function = get_cost_function(config),
        batch_size = 1,
    )
    trainer.test(model=model, datamodule=ptb)
    results_path = join(
        *(config["experiment"]["checkpoint_path"])[:-2], 
        f'{config["experiment"]["experiment_name"]}.pkl')
    df = pd.DataFrame(model.results)
    loss_mean = df["loss"].mean()
    print(f"Test loss: {loss_mean}")
    print(f"Test pplx: {math.exp(loss_mean)}")
    if dump_results:
        SequenceModelWrapper.dump_results(model.results, results_path)

def inference(
        config: dict[str, Any],
        prompt: str,
        mode: str = "argmax",
        lang_path: str = "lang.pkl",
        max_len: int = 30,
        allow_unk: bool = False,
    ):
    with open(lang_path, "rb") as f:
        lang = pickle.load(f)
    model = SequenceModelWrapper.load_model(
        checkpoint_path = join(*config["experiment"]["checkpoint_path"]),
        map_location = get_device(),
        model = get_model(config, len(lang.words2ids)),
        cost_function = get_cost_function(config),
    )

    temperatures = [0.5, 0.7, 0.75, 0.8, 1.0]
    for temp in temperatures:
        generated = model.generate(
            prompt, 
            lang, 
            mode=mode, 
            max_len=max_len,
            allow_unk=allow_unk,
            temperature=temp)
        print(f"t:{temp} => {generated}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Base interface")
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
        dest="dump_results", 
        help="Flag for dumping results object after test run"
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
        help="Path of inference configuration file"
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
    elif args.evaluate:
        evaluate(config, args.dump_results)
    elif args.inference:
        inference_config_path = args.inference_config_path | join("configs", "inference.yaml")
        inference_config = load_config(args.inference_config_path)
        inference(
            config, 
            prompt=args.prompt,
            mode=inference_config["mode"],
            max_len=inference_config["max_len"],
            allow_unk=inference_config["allow_unk"],
            lang_path=inference_config["lang_path"],
        )
    else:
        raise ValueError("Please provide a supported mode flag ('-t', '-e', '-i')")
    