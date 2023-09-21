from os.path import join
from argparse import ArgumentParser

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Base interface for training, evaluation and inference")
    parser.add_argument(
        "-c", 
        "--config", 
        type=str, 
        dest="config_path", 
        help="Path of configuration file"
    )
    parser.add_argument(
        "-t", 
        "--t", 
        action="store_true", 
        dest="train", 
        help="Flag for train mode"
    )
    parser.add_argument(
        "-e", 
        "--evaluate", 
        action="store_true", 
        dest="evaluate", 
        help="Flag for evaluation mode"
    )
    parser.add_argument(
        "-d", 
        "--dump-results", 
        action="store_true", 
        dest="dump_outputs", 
        help="Flag for dumping test outputs object after test run"
    )
    parser.add_argument(
        "-i", 
        "--inference", 
        action="store_true", 
        dest="inference", 
        help="Flag for inference mode"
    )
    parser.add_argument(
        "-ic", 
        "--inference-config", 
        type=str, 
        dest="inference_config_path", 
        default=join("configs", "inference.yaml"),
        help="Path of inference configuration file. Defaults to 'configs/inference.yaml''"
    )
    parser.add_argument(
        "-it", 
        "--interactive", 
        action="store_true", 
        default=False,
        help="Flag for interactive inference mode"
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        dest="prompt",
        default="the",
        help="Prompt for inference mode",
    )

    return parser