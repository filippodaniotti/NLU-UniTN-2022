import math
import logging
import pandas as pd
from os import makedirs
from os.path import join, isdir

import pytorch_lightning as pl

from .getters import get_data_module, get_logger, get_model, seed_everything, load_lang, get_device
from models import SequenceModelWrapper

from typing import Any

def train(config: dict[str, Any]):
    seed_everything(config["experiment"]["seed"])
    ptb = get_data_module(config)
    ptb.prepare_data() 
    logger = get_logger(config)
    trainer = pl.Trainer(max_epochs=config["experiment"]["epochs"], logger=logger)
    model = get_model(config, ptb.vocab_size, train=True)
    trainer.fit(model=model, datamodule=ptb)

def evaluate(config: dict[str, Any], dump_outputs: bool | None):
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