import os
from argparse import ArgumentParser
from datasets import load_dataset

from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl

from data import SentsDataset, get_collator
from lang import Lang
from models.lstm import BaseLSTM

def get_data():
    DS_PATH = "penn_treebank"
    ptb_train_path = "ptb.train.txt"
    ptb_test_path = "ptb.test.txt"
    ptb_valid_path = "ptb.valid.txt"
    batch_size = 64

    dataset = load_dataset(DS_PATH, data_files={"train": ptb_train_path, "test": ptb_test_path, "valid": ptb_valid_path})

    lang = Lang(dataset["train"]["text"], parse_sents=True)

    train_dataset = SentsDataset(dataset["train"]["text"], lang.words2ids)
    valid_dataset = SentsDataset(dataset["valid"]["text"], lang.words2ids)
    test_dataset = SentsDataset(dataset["test"]["text"], lang.words2ids)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=get_collator())
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=get_collator())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=get_collator())

    return lang, train_loader, valid_loader, test_loader

def train():
    emb_dim = 300
    hid_dim = 300
    lang, train_loader, valid_loader, test_loader = get_data() 
    # for v, t in enumerate(train_loader):
    #     print(v)
    # return 
    trainer = pl.Trainer()
    model = BaseLSTM(len(lang.words2ids), emb_dim, hid_dim)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

if __name__ == "__main__":
    parser = ArgumentParser(description="Base interface")
    parser.add_argument(
        "-t", "--train", action="store_true", help="Train model flag"
    )

    args = parser.parse_args()

    if args.train:
        train()