import sys
sys.path.insert(0, ".")
import pickle
from argparse import ArgumentParser
from data import PennTreebank

def main(filename: str) -> None:
    ptb = PennTreebank("https://data.deepai.org/ptbdataset.zip", "penn_treebank")
    ptb.prepare_data()
    with open(filename, "wb") as f:
        pickle.dump(ptb.lang, f)

if __name__ == "__main__":
    parser = ArgumentParser(description="Utility for dumping the Lang object")
    parser.add_argument(
        "-f", 
        "--filename", 
        type=str, 
        dest="filename", 
        default="lang.pkl",
        help="Path to save the lang object. Defaults to 'lang.pkl'"
    )

    args = parser.parse_args()
    main(args.filename)