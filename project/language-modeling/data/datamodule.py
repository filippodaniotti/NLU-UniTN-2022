import os
import wget
from zipfile import ZipFile

import datasets as hf_datasets
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader

from typing import Union, Any

from .lang import Lang
from .dataset import SentsDataset
from .collator import SequenceCollator

class PennTreebank(pl.LightningDataModule):
    def __init__(self,
            download_url: str,
            data_dir: str,
            temp_zip_name = "ptb.zip",
            batch_size: int = 64,
            tbptt: bool = False,
            tbptt_config: dict[str, Any] | None = None,
            pad_value: int = 0,):
        super().__init__()
        self.download_url = download_url
        self.data_dir = data_dir
        self.temp_zip_name = temp_zip_name
        self.batch_size = batch_size
        self.pad_value = pad_value
        self.tbptt = tbptt
        self.tbptt_config = tbptt_config

        # placeholders
        self.vocab_size: int = -1
        self.dataset: hf_datasets.DatasetDict | None = None
        self.lang: Lang | None = None
        self.ptb_train: SentsDataset | None = None
        self.ptb_val: SentsDataset | None = None
        self.ptb_test: SentsDataset | None = None

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

        self.dataset = hf_datasets.load_dataset(ds_path, data_files = {
            "train": "ptb.train.txt",
            "test":"ptb.test.txt",
            "valid": "ptb.valid.txt"})
        self.lang = Lang(self.dataset["train"]["text"], pad_value=self.pad_value, parse_sents=True)
        self.vocab_size = len(self.lang)

    def setup(self, stage: str):
        if stage == "fit":
            self.ptb_train = SentsDataset(self.dataset["train"]["text"], self.lang)
        if stage == "fit" or stage == "valid":
            self.ptb_val = SentsDataset(self.dataset["valid"]["text"], self.lang)
        if stage == "test":
            self.ptb_test = SentsDataset(self.dataset["test"]["text"], self.lang)

    def train_dataloader(self):
        return DataLoader(
            self.ptb_train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=SequenceCollator(
                pad_value = self.pad_value,
                tbptt = self.tbptt,
                tbptt_config = self.tbptt_config),
            num_workers=4)


    def val_dataloader(self):
        return DataLoader(
            self.ptb_val,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=SequenceCollator(),
            num_workers=4)

    def test_dataloader(self):
        return DataLoader(
            self.ptb_test,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=SequenceCollator(),
            num_workers=4)
