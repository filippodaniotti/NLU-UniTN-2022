from argparse import ArgumentParser

import pytorch_lightning as pl

from data.data_module import PennTreebank 
from models.lstm import BaselineLSTM
from models.wrapper import SequenceModel

from configs import EMBEDDING_DIM, HIDDEN_DIM, DS_URL, DS_PATH, BATCH_SIZE, EPOCHS, LOGS_PATH


def train(experiment_name: str):
    ptb = PennTreebank(DS_URL, DS_PATH, BATCH_SIZE)
    ptb.prepare_data()
    logger = pl.loggers.TensorBoardLogger(LOGS_PATH, experiment_name)
    trainer = pl.Trainer(max_epochs=EPOCHS, logger=logger)
    model = SequenceModel(BaselineLSTM(
        ptb.vocab_size,
        EMBEDDING_DIM,
        HIDDEN_DIM,
        num_layers=1,
        init_weights=True
        ))
    trainer.fit(model=model, datamodule=ptb)

if __name__ == "__main__":
    parser = ArgumentParser(description="Base interface")
    parser.add_argument(
        "-t", "--train", action="store_true", help="Train model flag"
    )
    parser.add_argument(
        "-e", "--experiment-name", action="store_true", help="Train model flag"
    )

    args = parser.parse_args()

    if args.train:
        train(args.experiment_name)