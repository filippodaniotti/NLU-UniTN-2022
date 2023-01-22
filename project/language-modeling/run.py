from argparse import ArgumentParser
from datasets import load_dataset

from torch.utils.data.dataloader import DataLoader

from data import SentsDataset, get_collator
from lang import Lang

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

    return train_loader, valid_loader, test_loader

def train():
    train_loader, valid_loader, test_loader = get_data() 

if __name__ == "__main__":
    parser = ArgumentParser(description="Base interface")
    parser.add_argument(
        "-t", "--train", action="store_true", help="Train model flag"
    )

    args = parser.parse_args()

    if args.train:
        train()