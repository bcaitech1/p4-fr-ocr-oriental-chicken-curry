import os
import wandb
import argparse
import multiprocessing
import numpy as np
import random
import time
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms
import yaml
from pprint import pprint
from tqdm import tqdm
from checkpoint import (
    default_checkpoint,
    load_checkpoint,
    save_checkpoint
)
from psutil import virtual_memory

from flags import Flags
from utils import get_network, get_optimizer, seed_everything, YamlConfigManager
from dataset import dataset_loader, START, PAD, load_vocab
from scheduler import CircularLRBeta, CosineAnnealingWarmupRestarts

from metrics import word_error_rate, sentence_acc


def main(config_file):
    options = Flags(config_file).get()
    cfg = YamlConfigManager(config_file)

    seed_everything(options.seed)

    is_cuda = torch.cuda.is_available()
    hardware = "cuda" if is_cuda else "cpu"
    device = torch.device(hardware)
    print("--------------------------------")
    print("Running {} on device {}\n".format(options.network, device))

    # Print system environments
    num_gpus = torch.cuda.device_count()
    num_cpus = os.cpu_count()
    mem_size = virtual_memory().available // (1024 ** 3)
    print(
        "[+] System environments\n",
        "The number of gpus : {}\n".format(num_gpus),
        "The number of cpus : {}\n".format(num_cpus),
        "Memory Size : {}G\n".format(mem_size),
    )

    # Load checkpoint and print result
    checkpoint = (
        load_checkpoint(options.checkpoint, cuda=is_cuda)
        if options.checkpoint != ""
        else default_checkpoint
    )
    model_checkpoint = checkpoint["model"]
    if model_checkpoint:
        print(
            "[+] Checkpoint\n",
            "Resuming from epoch : {}\n".format(checkpoint["epoch"]),
            "Train Symbol Accuracy : {:.5f}\n".format(checkpoint["train_symbol_accuracy"][-1]),
            "Train Sentence Accuracy : {:.5f}\n".format(checkpoint["train_sentence_accuracy"][-1]),
            "Train WER : {:.5f}\n".format(checkpoint["train_wer"][-1]),
            "Train Loss : {:.5f}\n".format(checkpoint["train_losses"][-1]),
            "Validation Symbol Accuracy : {:.5f}\n".format(
                checkpoint["validation_symbol_accuracy"][-1]
            ),
            "Validation Sentence Accuracy : {:.5f}\n".format(
                checkpoint["validation_sentence_accuracy"][-1]
            ),
            "Validation WER : {:.5f}\n".format(
                checkpoint["validation_wer"][-1]
            ),
            "Validation Loss : {:.5f}\n".format(checkpoint["validation_losses"][-1]),
        )

    # Get data
    transformed = A.Compose([
        A.Resize(options.input_size.height, options.input_size.width),
        A.OneOf([
            A.Rotate(),
            A.ShiftScaleRotate(),
            A.RandomRotate90(),
            A.VerticalFlip()
        ]),
        A.OneOf([
            A.MotionBlur(),
            A.Blur(),
            A.GaussianBlur()
        ]),
        ToTensorV2()
    ])    

    train_data_loader, validation_data_loader, train_dataset, valid_dataset = dataset_loader(options, transformed)
    print(
        "[+] Data\n",
        "The number of train samples : {}\n".format(len(train_dataset)),
        "The number of validation samples : {}\n".format(len(valid_dataset)),
        "The number of classes : {}\n".format(len(train_dataset.token_to_id)),
    )

    pprint(options.network)
    pprint(options)

    model = get_network(
        options.network,
        options,
        model_checkpoint,
        device,
        train_dataset,
    )

    summary(model)
    inp = torch.zeros((2, 1, 128, 256)).cuda()
    out = model.encoder(inp)
    print(out.shape)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_file",
        dest="config_file",
        default="./configs/SATRN.yaml",
        type=str,
        help="Path of configuration file",
    )
    parser = parser.parse_args()    
    main(parser.config_file)