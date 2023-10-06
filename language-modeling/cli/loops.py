import math
import logging
import pandas as pd
from os import makedirs
from os.path import join, isdir

import pytorch_lightning as pl

from .getters import get_data_module, get_logger, get_model, seed_everything, load_lang, get_device
from models import SequenceModelWrapper

from typing import Any

def train(config: dict[str, Any]) -> None:
    """
    Train a PyTorch Lightning model using the provided configuration.

    Args:
        config (dict[str, Any]): Dictionary containing the full experiment configuration.
    """
    seed_everything(config["experiment"]["seed"])
    ptb = get_data_module(config)
    ptb.prepare_data() 
    callbacks = []
    if config["experiment"].get("early_stopping", 0) > 0:
        callbacks.append(pl.callbacks.EarlyStopping(
            monitor="Loss/Valid", 
            mode="min", 
            patience=config["experiment"]["early_stopping"]))
    trainer = pl.Trainer(
        max_epochs=config["experiment"]["epochs"], 
        logger=get_logger(config), 
        callbacks=callbacks)
    model = get_model(config, ptb.vocab_size, train=True)
    trainer.fit(model=model, datamodule=ptb)

def evaluate(config: dict[str, Any], dump_outputs: bool | None) -> pd.DataFrame:
    """
    Evaluate a PyTorch Lightning model using the provided configuration.
    It behaviors differently depending on the value of `dump_outputs`:
        - If `dump_outputs` is `False`, the model is evaluated on both
            the validation and test sets and no output is dumped.
        - If `dump_outputs` is `True`, the evaluation epoch is performed
            only on the test set andthe model outputs are dumped to a file.

    Args:
        config (dict[str, Any]): Dictionary containing the full experiment configuration.
        dump_outputs (bool | None): Whether to dump the model outputs to a file.

    Returns:
        pd.DataFrame: Dataframe containing the evaluation metrics.
    """
    # disable device information logging
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    ptb = get_data_module(config, batch_size=1)
    ptb.prepare_data()
    trainer = pl.Trainer(logger=False, enable_model_summary=False)
    model = get_model(config, ptb.vocab_size, train=False)

    def _run_loop(split: str, split_fn: callable):
        ptb.setup(split)
        split_fn(model=model, datamodule=ptb, verbose=False)
        loss_mean = float(pd.DataFrame(model.outputs)["loss"].mean())
        split = split[0].upper() + split[1:]
        metrics[split] = [loss_mean, math.exp(loss_mean)]

    metrics = {}
    if not dump_outputs: 
        _run_loop("valid", trainer.validate)
    _run_loop("test", trainer.test)

    if dump_outputs:
        outputs_path = join(config["results"]["results_path"], "outputs")
        if not isdir(config["results"]["results_path"]) or not isdir(outputs_path):
            makedirs(outputs_path)
        results_path = join(outputs_path, f'{config["experiment"]["experiment_name"]}.pkl')
        SequenceModelWrapper.dump_outputs(model.outputs, results_path)

    return pd.DataFrame(metrics, index=["Loss", "Perplexity"])

def inference_step(
        model: SequenceModelWrapper, 
        prompt: str, 
        lang: dict[str, int], 
        inf_config: dict[str, Any],
        temperature: float,
        device: str
    ) -> str:
    """
    Wrapper function for the inference step.

    Args:
        model (SequenceModelWrapper): The model to be used for inference. 
        prompt (str): The prompt to be fed to the model
        lang (dict[str, int]): The Lang object to be used for the word mappings.
        inf_config (dict[str, Any]): The inference configuration
        temperature (float): The temperature to be used for the logits smoothing.
        device (str): The device on which to load the model.

    Returns:
        str: The generated sequence.
    """
    return model.generate(
        prompt, 
        lang, 
        mode=inf_config["mode"], 
        max_len=inf_config["max_length"],
        allow_unk=inf_config["allow_unk"],
        temperature=temperature,
        device=device)

def inference(
        config: dict[str, Any], 
        inf_config: dict[str, Any], 
        prompt: str
    ) -> str:
    """
    Standard inference mode. It uses the model to generate n sequences,
    where n is the number of temperature values provided in the
    inf_config dictionary.

    Args:
        config (dict[str, Any]): Dictionary containing the full experiment configuration.
        inf_config (dict[str, Any]): The inference configuration
        prompt (str): The prompt to be fed to the model.

    Returns:
        str: The generated sequence(s).
    """
    lang = load_lang(join(*inf_config["lang_path"]))
    model = get_model(config, len(lang), train=False)

    out = ""
    for temp in inf_config["temperatures"]:
        generated = inference_step(model, prompt, lang, inf_config, temp, get_device())
        out += f"T: {temp:.2f}\t=>\t{generated}\n"
        
    return out