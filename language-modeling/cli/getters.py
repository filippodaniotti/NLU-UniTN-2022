import yaml
import pickle
import numpy as np
from os import listdir
from os.path import join, isfile

import torch
import torch.nn as nn
import pytorch_lightning as pl

from data import PennTreebank, Lang
from models import BaselineLSTM, MerityLSTM, SequenceModelWrapper

from typing import Any

def load_config(config_path: str) -> dict[str, Any]:
    """
    Load a configuration file from a given path.

    Args:
        config_path (str): The path of the configuration file.

    Returns:
        dict[str, Any]: The loaded configuration dictionary
    """
    with open(config_path) as config_file:
        return yaml.safe_load(config_file)

def load_lang(lang_path: str) -> Lang:
    """
    Load a dumped Lang object from a given path.
    See the tools/dump_lang.py utility for more information
    on how to dump the Lang object.

    Args:
        lang_path (str): The path of the dumped Lang object.

    Returns:
        Lang: The loaded Lang object.
    """
    with open(lang_path, "rb") as lang_file:
        return pickle.load(lang_file)

def get_device() -> torch.device:
    """
    Get device given the host configuration.

    Returns:
        torch.device: The available device(s).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed: int) -> None:
    """
    Wrapper for Numpy, PyTorch and Pytorch Lightning seeding utility.
    We use seeding to ensure reproducibility of the experiments.

    Args:
        seed (int): The seed to be used.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed)

def get_weights_path(config: dict[str, Any]) -> str:
    """
    Get the path to the weights checkpoint file based on the provided 
    configuration. This function retrieves the path to the weights 
    checkpoint file for a given experiment based on the provided
    configuration. It checks two potential sources for the checkpoint path,
    in the following order of precedence:

    1. If the 'checkpoint_path' key is present in the 'experiment' section of the 
        configuration, it returns the path specified by that key.
    2. If the 'checkpoint_path' key is not provided, it checks for the checkpoint file 
        in the following locations in order:
        a. The 'results_path' specified in the 'results' section of the configuration, under a 'weights' subdirectory,
            with the filename '{experiment_name}.ckpt'.
        b. If not found in (a), it checks in the 'logs_path' specified in the 'results' section of the configuration,
            under a directory structure like 'experiment_name/version_0/checkpoints', and returns the first '.ckpt' file
            found in that directory.

    If no checkpoint file is found in any of the specified locations, a ValueError is raised.

    Args:
        config (dict[str, Any]): A dictionary containing configuration parameters.

    Raise:
        ValueError: No valid checkpoint could be found with the provided information

    Returns:
        str: The path to the weights checkpoint file.

    Note:
    - This function assumes that the specified paths and filenames are consistent with the actual directory structure
      and naming conventions used in your project.
    """
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
    """
    Getter for the nn.Module inner model to be used in the experiment.

    Args:
        config (dict[str, Any]): Dictionary containing the full experiment configuration.
        vocab_size (int): The Vocab size, which would be the number of classes in the output space.

    Raises:
        ValueError: No valid model was provided. Either 'baseline' or 'merity' should be used.

    Returns:
        nn.Module: The nn.Module of the inner model.
    """
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
        raise ValueError(f"Provided model '{config['experiment']['model']}' not available.")

def get_model(
        config: dict[str, Any], 
        vocab_size: int,
        train: bool = True,) -> SequenceModelWrapper:
    """
    Get a SequenceModelWrapper instance based on the provided configuration 
    and vocabulary size.

    This function returns a sequence model based on the provided configuration 
    and vocabulary size. The behavior of the function depends on the 'train' flag:
        - If 'train' is True, it returns a new training-ready model wrapped with 
            training-specific parameters, including the model architecture, cost function, 
            optimizer, learning rate, and other training-related settings.

        - If 'train' is False, it loads a pre-trained model from the checkpoint 
            file specified in the configuration and returns it for evaluation. 
            It also sets the model architecture and cost function.

    Args:
        config (dict[str, Any]): Dictionary containing the full 
            experiment configuration.
        vocab_size (int): The size of the vocabulary for the 
            sequence model, i.e. the number of classes.
        train (bool, optional): A flag indicating whether the model 
            is for training or evaluation. Defaults to True.

    Returns:
        SequenceModelWrapper: The SequenceModelWrapper instance, created 
            with training or evaluation-specific parameters.

    Note:
    - This function assumes that the necessary model architecture, cost function, 
        and other parameters are defined in a properly-constructed configuration file
        according to the conventions of this project.
    - When 'train' is False, the function also assumes that the checkpoint path is 
        correctly specified in the configuration file and that a valid pre-trained 
        model exists at that location.
    """
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
        batch_size: int | None = None
    ) -> PennTreebank:
    """
    Getter for the PennTreebank DataModule class.

    Args:
        config (dict[str, Any]): Dictionary containing the full experiment configuration.
        batch_size (int | None, optional): Optional custom batch. 
            This is set to 1 when run in Evaluation or Inference mode. Defaults to None.

    Returns:
        PennTreebank: The PennTreebank DataModule class.
    """
    batch_size = batch_size or config["experiment"]["batch_size"]
    return PennTreebank(
        download_url = config["dataset"]["ds_url"],
        data_dir = config["dataset"]["ds_path"],
        pad_value = config["dataset"]["pad_value"],
        part_shuffle = config["experiment"].get("part_shuffle", False),
        tbptt = bool(config["experiment"].get("tbptt", False)),
        tbptt_config = config["experiment"].get("tbptt_config", None),
        batch_size = batch_size,
    )

def get_logger(config: dict[str, Any]) -> pl.loggers.TensorBoardLogger:
    """
    Getter for the TensorBoardLogger for the experiment.

    Args:
        config (dict[str, Any]): Dictionary containing the full experiment configuration.

    Returns:
        pl.loggers.TensorBoardLogger: The TensorBoardLogger instance.
    """
    return pl.loggers.TensorBoardLogger(
        config["results"]["logs_path"],
        config["experiment"]["experiment_name"]
    )
    
def get_cost_function(config: dict[str, Any]) -> nn.Module:
    """
    Getter for the cost function for the experiment.
    In this task, we are using the CrossEntropy loss function.
    We are ignoring the index of the pad token from the loss computation.

    Args:
        config (dict[str, Any]): Dictionary containing the full experiment configuration.

    Returns:
        nn.Module: The CrossEntropyLoss nn.Module.
    """
    return nn.CrossEntropyLoss(ignore_index=config["dataset"]["pad_value"])

