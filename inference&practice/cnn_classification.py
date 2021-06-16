import os
import argparse
import numpy as np
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms
import yaml
from tqdm import tqdm
from psutil import virtual_memory
from flags import Flags
from utils import get_network, get_optimizer, seed_everything, YamlConfigManager
from datasets import dataset_loader
from scheduler import CircularLRBeta, CosineAnnealingWarmupRestarts, StepLR
from networks.nfnet import nfnet
from albumentations import ImageOnlyTransform
import cv2


class Denoising(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        return cv2.fastNlMeansDenoising(img, h=5)


def run_epoch(
    data_loader,
    model,
    optimizer,
    lr_scheduler,
    device,
    train=True,
):
    # Disables autograd during validation mode
    torch.cuda.empty_cache()
    torch.set_grad_enabled(train)
    criterion = nn.CrossEntropyLoss()
    if train:
        model.train()
    else:
        model.eval()
    losses = []
    class_acc = []
    model.to(device)
    with tqdm(data_loader, total=len(data_loader.dataset), unit='batch') as pbar:
        for d in data_loader:
            image = d["image"].float().to(device)
            class_label = d['label'].type(torch.long).to(device)
            # The last batch may not be a full batch
            output = model(image)
            if train:
                loss = criterion(output, class_label)
            pred = torch.argmax(output, dim=1)
            class_acc.append(np.sum((pred.detach().cpu().numpy(
            ) == class_label.detach().cpu().numpy()))/len(image))
            if train:
                optimizer.zero_grad()
                loss.backward()
                # cycle
                optimizer.step()
                lr_scheduler.step()

            losses.append(loss.item())
            del loss
            del output
            pbar.update(len(image))
            pbar.set_postfix(run_loss=np.mean(losses),
                             run_acc=np.mean(class_acc))
    result = {
        "loss": np.mean(losses),
        'ACC': np.mean(class_acc)
    }

    return result


def main(config_file):
    """
    Train math formula recognition model
    """
    options = Flags(config_file).get()
    cfg = YamlConfigManager(config_file)
    # set wandb
    #set random seed
    seed_everything(options.seed)
    is_cuda = torch.cuda.is_available()
    hardware = "cuda" if is_cuda else "cpu"
    device = torch.device(hardware)

    # Get data

    train_transform = A.Compose([
        A.Resize(options.input_size.height, options.input_size.width),
        A.OneOf([
            A.GaussianBlur(),
            A.CLAHE(),
            A.MotionBlur()
        ], p=0.5),
        Denoising(p=1.0),
    ])

    valid_transform = A.Compose([
        A.Resize(options.input_size.height, options.input_size.width),
        Denoising(p=1.0),
    ])

    train_data_loader, validation_data_loader, train_dataset, valid_dataset = dataset_loader(
        options, train_transform, valid_transform)

    # Get loss, model
    model = nfnet()
    model.train()
    # Get optimizer
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    if options.optimizer.scheduler == 'step':
      lr_scheduler = StepLR(optimizer=optimizer,
                            step_size=1000, verbose=False, last_epoch=-1)

    # Train
    best_acc = 0.0
    for epoch in range(options.num_epochs):
        start_time = time.time()
        # Train
        train_result = run_epoch(
            train_data_loader,
            model,
            optimizer,
            lr_scheduler,
            device,
            train=True
        )
        epoch_lr = lr_scheduler.get_last_lr()[0]  # cycle

        # Validation
        validation_result = run_epoch(
            validation_data_loader,
            model,
            optimizer,
            lr_scheduler,
            options.max_grad_norm,
            device,
            train=False,
        )
        if validation_result['ACC'] > best_acc:
          best_acc = validation_result['ACC']
          best_model = copy.deepcopy(model.state.dict())
          model_path = './log/best.pth'
          torch.save(best_model, model.path)
          print('best_model_saved')


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
