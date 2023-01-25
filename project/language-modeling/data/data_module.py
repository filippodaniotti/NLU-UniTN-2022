import os
import wget
from zipfile import ZipFile

import datasets
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader

from typing import Union

from .lang import Lang
from .custom_datasets import SentsDataset
from .collators import get_collator

class PennTreebank(pl.LightningDataModule):
    def __init__(self, 
            download_url: str,
            data_dir: str, 
            temp_zip_name = "ptb.zip",
            batch_size: int = 64):
            
        super().__init__()
        self.download_url = download_url
        self.data_dir = data_dir
        self.temp_zip_name = temp_zip_name
        self.batch_size = batch_size

        # placeholders
        self.vocab_size: int = -1
        self.dataset = None
        self.lang: Union[Lang, None] = None
        self.ptb_train: Union[SentsDataset, None] = None
        self.ptb_val: Union[SentsDataset, None] = None
        self.ptb_test: Union[SentsDataset, None] = None

    def _download_and_extract(self, ds_path: str, zip_path: str) -> None:
        if not os.path.isdir(ds_path):
            if not os.path.isfile(zip_path):
                print("Downloading...")
                wget.download(self.download_url, zip_path)
            with ZipFile(zip_path, "r") as zip_ref:
                print("Extracting...")
                zip_ref.extractall(ds_path)


    def prepare_data(self) -> None:
        # download
        temp_path = os.path.join(os.getcwd(), self.temp_zip_name)
        ds_path = os.path.join(os.getcwd(), self.data_dir)
        self._download_and_extract(ds_path, temp_path)

        ptb_train_path = "ptb.train.txt"
        ptb_test_path = "ptb.test.txt"
        ptb_valid_path = "ptb.valid.txt"

        self.dataset = datasets.load_dataset(ds_path, data_files = {
            "train": ptb_train_path, 
            "test": ptb_test_path, 
            "valid": ptb_valid_path})
        self.lang = Lang(self.dataset["train"]["text"], parse_sents=True)
        self.vocab_size = len(self.lang.words2ids)

    def setup(self, stage: str) -> None:
        self.ptb_train = SentsDataset(self.dataset["train"]["text"], self.lang.words2ids)
        self.ptb_val = SentsDataset(self.dataset["valid"]["text"], self.lang.words2ids)
        self.ptb_test = SentsDataset(self.dataset["test"]["text"], self.lang.words2ids)

    def train_dataloader(self):
        return DataLoader(self.ptb_train, batch_size=self.batch_size, shuffle=True, collate_fn=get_collator())

    def val_dataloader(self):
        return DataLoader(self.ptb_val, batch_size=self.batch_size, shuffle=True, collate_fn=get_collator())

    def test_dataloader(self):
        return DataLoader(self.ptb_test, batch_size=self.batch_size, shuffle=True, collate_fn=get_collator())

    # def predict_dataloader(self):
        # return DataLoader(self.mnist_predict, batch_size=self.batch_size)

    # def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...