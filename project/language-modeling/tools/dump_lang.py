import sys
sys.path.append(".")
import pickle
from argparse import ArgumentParser
from data.data_module import PennTreebank

def main(filename: str) -> None:
    ptb = PennTreebank("https://data.deepai.org/ptbdataset.zip", "penn_treebank")
    ptb.prepare_data()
    with open(filename, "wb") as f:
        pickle.dump(ptb.lang, f)

if __name__ == "__main__":
    parser = ArgumentParser(description="Base interface")
    parser.add_argument(
        "-f", 
        "--filename", 
        type=str, 
        dest="filename", 
        help="Path to save the lang object"
    )

    args = parser.parse_args()
    main(args.filename)