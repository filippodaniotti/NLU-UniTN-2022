"""
Expects to parse a results folder that looks like this:
./results
| ---- experiment_name
|      | ---- version_0
|      |      | ---- events.out.tfevents.(id)(test_before_training)
|      |      | ---- events.out.tfevents.(id)(training)
|      |      | ---- events.out.tfevents.(id)(test_after_training)
|      |      | ---- checkpoints
|      |      |      | ---- epoch=(last_epoch)-steps=(last_step).ckpt
|      |      | ---- useless stuff
"""

import os
from argparse import ArgumentParser
from shutil import rmtree, copy, copyfile
from subprocess import run

from typing import List

import tflogs2pandas

weights_dir = "weights"
tblog_dir = "tensorboard"
csv_dir = "csv"

def make_tree(path: str, exp_name: str):
    exp_path = os.path.join(path, exp_name)
    os.makedirs(exp_path)
    return exp_path

def filter_tb_logs(log_list: List[str]) -> List[str]:
    return list(filter(lambda file: "tfevents" in file, log_list))

def copy_tb(exp_dir: str, exp_name: str, tb_path: str):
    exp_path = make_tree(tb_path, exp_name)
    logs = os.listdir(exp_dir)
    logs = filter_tb_logs(logs)
    for file in logs:
        copy(os.path.join(exp_dir, file), exp_path)

def copy_ckpt(exp_dir: str, exp_name: str, weights_path: str):
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    ckpt = os.listdir(ckpt_dir)[0]
    copyfile(os.path.join(ckpt_dir, ckpt), os.path.join(weights_path, f"{exp_name}.ckpt"))

def convert_to_csv(exp_name: str, csv_path: str):
    exp_path = os.path.join(csv_path, exp_name)
    tb_logs = exp_path.replace("csv", "tensorboard")
    tflogs2pandas.main(tb_logs, False, True, csv_path)
    filename = "all_training_logs_in_one_file.csv"
    os.rename(os.path.join(csv_path, filename), os.path.join(csv_path, f"{exp_name}.csv"))
    


def main(logs_dir: str, results_dir: str):
    dirs = os.listdir(logs_dir)
    if os.path.isdir(results_dir):
        rmtree(results_dir)
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

        weights_path = os.path.join(results_dir, weights_dir)
        tb_path = os.path.join(results_dir, tblog_dir)
        csv_path = os.path.join(results_dir, csv_dir)
        
        os.makedirs(weights_path)
        os.makedirs(tb_path)
        os.makedirs(csv_path)

    for exp in dirs:
        exp_dir = os.path.join(logs_dir, exp, "version_0")
        # copy tensorboard logs
        copy_tb(exp_dir, exp, tb_path)
        copy_ckpt(exp_dir, exp, weights_path)
        convert_to_csv(exp, csv_path)

        

if __name__ == "__main__":
    parser = ArgumentParser(description="Pytorch lightning logs parser")
    parser.add_argument(
        "-l", 
        "--logs-directory", 
        dest="logs_dir", 
        help="Name of logs directory",
        required=True
    )
    parser.add_argument(
        "-d", 
        "--dest-directory", 
        dest="dest", 
        help="Name of the destination directory for results",
        required=True
    )

    args = parser.parse_args()
    main(args.logs_dir, args.dest)
    