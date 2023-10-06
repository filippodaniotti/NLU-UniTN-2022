from os.path import join
from argparse import ArgumentParser

from torch.multiprocessing import set_sharing_strategy
set_sharing_strategy('file_system')

from cli import launch_tui, train, evaluate, inference, load_config

if __name__ == "__main__":
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

    args = parser.parse_args()
    config = load_config(args.config_path)

    if args.train:
        train(config)
    if args.evaluate:
        metrics = evaluate(config, args.dump_outputs)
        print(metrics)
    if args.inference:
        inference_config = load_config(args.inference_config_path)
        if not args.interactive:
            generated = inference(config, inference_config, args.prompt)
            print(generated)
        else:
            print("Launching TUI application...")
            launch_tui(config, inference_config)
            print("Quitting TUI application...")
    if not any([args.train, args.evaluate, args.inference]):
        raise ValueError("Please provide a supported mode flag ('-t', '-e', '-i')")
    