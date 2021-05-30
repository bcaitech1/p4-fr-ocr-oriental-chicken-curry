import os
import argparse
import multiprocessing
import wandb
import logging
import random
import time
import shutil
import yaml

import numpy as np
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm

from psutil import virtual_memory

from utils.flags import Flags
from utils.utils import get_network, get_optimizer
from utils.metrics import word_error_rate,sentence_acc
from utils.scheduler import CircularLRBeta,CosineAnnealingWarmUpRestarts
from utils.checkpoint import (
    default_checkpoint,
    load_checkpoint,
    save_checkpoint,
    init_tensorboard,
    write_tensorboard,
)

from data.dataset import dataset_loader, START, PAD,load_vocab
from data.augmentation import *
from log import *

def seed_everything(seed: int):
    """ 시드 고정 함수

    Args:
        seed(int) : 시드 고정값
    """

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def id_to_string(tokens, data_loader,do_eval=0):
    """token id 를 문자열로 변환하는 로직

    Args:
        tokens(list) : 토큰 아이디
        data_loader(Dataloaer) : 현재 사용하고 있는 데이터 로더
        do_eval(int): 0 - train, 이 외 - eval
    """
    result = []
    if do_eval:
        special_ids = [data_loader.dataset.token_to_id["<PAD>"], data_loader.dataset.token_to_id["<SOS>"],
                       data_loader.dataset.token_to_id["<EOS>"]]

    for example in tokens:
        string = ""
        if do_eval:
            for token in example:
                token = token.item()
                if token not in special_ids:
                    if token != -1:
                        string += data_loader.dataset.id_to_token[token] + " "
        else:
            for token in example:
                token = token.item()
                if token != -1:
                    string += data_loader.dataset.id_to_token[token] + " "

        result.append(string)
    return result

def run_epoch(
    epoch,
    ratio_cycle,
    data_loader,
    model,
    epoch_text,
    criterion,
    optimizer,
    lr_scheduler,
    teacher_forcing_ratio,
    max_grad_norm,
    device,
    model_type,
    train=True,
):
    # Disables autograd during validation mode
    torch.set_grad_enabled(train)
    if train:
        model.train()
    else:
        model.eval()

    losses = []
    grad_norms = []
    correct_symbols = 0
    total_symbols = 0
    wer=0
    num_wer=0
    sent_acc=0
    num_sent_acc=0

    with tqdm(
        desc="{} ({})".format(epoch_text, "Train" if train else "Validation"),
        total=len(data_loader.dataset),
        dynamic_ncols=True,
        leave=False,
    ) as pbar:
        for d in data_loader:
            input = d["image"].float().to(device)
            input = input.unsqueeze(1)

            # The last batch may not be a full batch
            curr_batch_size = len(input)
            expected = d["truth"]["encoded"].to(device)

            # Replace -1 with the PAD token
            # -1로 되어 있는 부분을 PAD Token 아이디로 대체
            expected[expected == -1] = data_loader.dataset.token_to_id[PAD]
            
            if model_type == "Attention":
                output = model(input, expected, train, teacher_forcing_ratio, epoch, ratio_cycle)
            elif model_type == "SRN":
                output = model(input)
            elif model_type == "SATRN":
                output = model(input, expected, train, teacher_forcing_ratio)
            
            decoded_values = output.transpose(1, 2)
            _, sequence = torch.topk(decoded_values, 1, dim=1)
            sequence = sequence.squeeze(1)
            
            loss = criterion(decoded_values, expected[:, 1:])

            if train:
                optim_params = [
                    p
                    for param_group in optimizer.param_groups
                    for p in param_group["params"]
                ]
                optimizer.zero_grad()
                loss.backward()
                # Clip gradients, it returns the total norm of all parameters
                grad_norm = nn.utils.clip_grad_norm_(
                    optim_params, max_norm=max_grad_norm
                )
                grad_norms.append(grad_norm)

                # cycle
                lr_scheduler.step()
                optimizer.step()

            if train:
                # # step base wandb log
                wandb.log(
                    {
                        "learning_rate": optimizer.param_groups[0]['lr']
                    }
                )

            losses.append(loss.item())
            
            expected[expected == data_loader.dataset.token_to_id[PAD]] = -1
            expected_str = id_to_string(expected, data_loader,do_eval=1)
            sequence_str = id_to_string(sequence, data_loader,do_eval=1)
            wer += word_error_rate(sequence_str,expected_str)
            num_wer += 1
            sent_acc += sentence_acc(sequence_str,expected_str)
            num_sent_acc += 1
            correct_symbols += torch.sum(sequence == expected[:, 1:], dim=(0, 1)).item()
            total_symbols += torch.sum(expected[:, 1:] != -1, dim=(0, 1)).item()

            pbar.update(curr_batch_size)

    expected = id_to_string(expected, data_loader)
    sequence = id_to_string(sequence, data_loader)
    print("-" * 10 + "GT ({})".format("train" if train else "valid"))
    print(*expected[:3], sep="\n")
    print("-" * 10 + "PR ({})".format("train" if train else "valid"))
    print(*sequence[:3], sep="\n")

    result = {
        "loss": np.mean(losses),
        "correct_symbols": correct_symbols,
        "total_symbols": total_symbols,
        "wer": wer,
        "num_wer":num_wer,
        "sent_acc": sent_acc,
        "num_sent_acc":num_sent_acc
    }
    if train:
        try:
            result["grad_norm"] = np.mean([tensor.cpu() for tensor in grad_norms])
        except:
            result["grad_norm"] = np.mean(grad_norms)

    return result


def main(config_file):
    """main logic수행

    Args:
        config_file(ArgumentParser) : configuration파일 정보를 가지고 있는 객체
    """
    # config 옵션 flag화
    options = Flags(config_file).get()

    # 시드 고정
    seed_everything(options.seed)

    # device설정
    device = get_device()

    # 시스템 환경 로그 출력
    get_enviroments_log()

    # config파일에 체크포인트 모델의 경로 지정시 모델을 가져옴
    checkpoint = (
        load_checkpoint(options.checkpoint, cuda=torch.cuda.is_available())
        if options.checkpoint != ""
        else default_checkpoint
    )
    model_checkpoint = checkpoint["model"]
    checkpoint_model_log(model_checkpoint,checkpoint)

    # Get data
    # 데이터셋을 받아옴
    train_data_loader, validation_data_loader, train_dataset, valid_dataset, _ = get_dataset(options)

    one_epoch_step = int(len(train_data_loader) / options.batch_size)

    # Get loss, model
    model = get_network(
        options.network,
        options,
        model_checkpoint,
        device,
        train_dataset,
    )
    model.train()

    criterion = model.criterion.to(device)
    enc_params_to_optimise = [
        param for param in model.encoder.parameters() if param.requires_grad
    ]
    dec_params_to_optimise = [
        param for param in model.decoder.parameters() if param.requires_grad
    ]
    params_to_optimise = [*enc_params_to_optimise, *dec_params_to_optimise]
    opt_param_log(options,enc_params_to_optimise,dec_params_to_optimise)

    # Get optimizer
    optimizer = get_optimizer(
        options.optimizer.optimizer,
        params_to_optimise,
        lr=options.optimizer.lr,
        weight_decay=options.optimizer.weight_decay,
    )

    optimizer_state = checkpoint.get("optimizer")
    if optimizer_state:# optimizer 설정이 이미 있으면
        optimizer.load_state_dict(optimizer_state)
    for param_group in optimizer.param_groups:
        param_group["initial_lr"] = options.optimizer.lr


    if options.optimizer.type == "cawr":# 스케줄러 조정
        lr_scheduler = CosineAnnealingWarmUpRestarts(
            optimizer,
            T_0=len(train_data_loader) * options.num_epochs,
            T_up=len(train_data_loader) * (options.num_epochs/2),
            T_mult=2,
            eta_max= 5e-4,
            gamma= 0.5,
            )
    elif options.optimizer.type == "cycle":
        cycle = len(train_data_loader) * options.num_epochs
        lr_scheduler = CircularLRBeta(
            optimizer, options.optimizer.lr, 10, 10, cycle, [0.95, 0.85]
        )
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=options.optimizer.lr_epochs,
            gamma=options.optimizer.lr_factor,
        )

    # Log
    if not os.path.exists(options.prefix):
        os.makedirs(options.prefix)
    # log_file = open(os.path.join(options.prefix, "log.txt"), "w")
    shutil.copy(config_file, os.path.join(options.prefix, "train_config.yaml"))
    if options.print_epochs is None:
        options.print_epochs = options.num_epochs
    # writer = init_tensorboard(name=options.prefix.strip("-"))
    start_epoch = checkpoint["epoch"]
    train_symbol_accuracy = checkpoint["train_symbol_accuracy"]
    train_sentence_accuracy=checkpoint["train_sentence_accuracy"]
    train_wer=checkpoint["train_wer"]
    train_losses = checkpoint["train_losses"]
    validation_symbol_accuracy = checkpoint["validation_symbol_accuracy"]
    validation_sentence_accuracy=checkpoint["validation_sentence_accuracy"]
    validation_wer=checkpoint["validation_wer"]
    validation_losses = checkpoint["validation_losses"]
    learning_rates = checkpoint["lr"]
    grad_norms = checkpoint["grad_norm"]

    # Train
    for epoch in range(options.num_epochs):
        start_time = time.time()

        # lr_scheduler.step()

        epoch_text = "[{current:>{pad}}/{end}] Epoch {epoch}".format(
            current=epoch + 1,
            end=options.num_epochs,
            epoch=start_epoch + epoch + 1,
            pad=len(str(options.num_epochs)),
        )

        # # Train
        train_result = run_epoch(
            epoch,
            options.ratio_cycle,
            train_data_loader,
            model,
            epoch_text,
            criterion,
            optimizer,
            lr_scheduler,
            options.teacher_forcing_ratio,
            options.max_grad_norm,
            device,
            model_type = options.network,
            train=True,
        )

        train_losses.append(train_result["loss"])
        grad_norms.append(train_result["grad_norm"])
        train_epoch_symbol_accuracy = (
            train_result["correct_symbols"] / train_result["total_symbols"]
        )
        train_symbol_accuracy.append(train_epoch_symbol_accuracy)
        train_epoch_sentence_accuracy = (
                train_result["sent_acc"] / train_result["num_sent_acc"]
        )

        train_sentence_accuracy.append(train_epoch_sentence_accuracy)
        train_epoch_wer = (
                train_result["wer"] / train_result["num_wer"]
        )
        train_wer.append(train_epoch_wer)
        epoch_lr = lr_scheduler.get_lr()  # cycle

        # Validation
        validation_result = run_epoch(
            epoch,
            options.ratio_cycle,
            validation_data_loader,
            model,
            epoch_text,
            criterion,
            optimizer,
            lr_scheduler,
            options.teacher_forcing_ratio,
            options.max_grad_norm,
            device,
            model_type = options.network,
            train=False,
        )
        validation_losses.append(validation_result["loss"])
        validation_epoch_symbol_accuracy = (
            validation_result["correct_symbols"] / validation_result["total_symbols"]
        )
        validation_symbol_accuracy.append(validation_epoch_symbol_accuracy)

        validation_epoch_sentence_accuracy = (
            validation_result["sent_acc"] / validation_result["num_sent_acc"]
        )
        validation_sentence_accuracy.append(validation_epoch_sentence_accuracy)
        validation_epoch_wer = (
                validation_result["wer"] / validation_result["num_wer"]
        )
        validation_wer.append(validation_epoch_wer)

        # Save checkpoint
        #make config
        with open(config_file, 'r') as f:
            option_dict = yaml.safe_load(f)

        save_checkpoint(
            {
                "epoch": start_epoch + epoch + 1,
                "train_losses": train_losses,
                "train_symbol_accuracy": train_symbol_accuracy,
                "train_sentence_accuracy": train_sentence_accuracy,
                "train_wer":train_wer,
                "validation_losses": validation_losses,
                "validation_symbol_accuracy": validation_symbol_accuracy,
                "validation_sentence_accuracy":validation_sentence_accuracy,
                "validation_wer":validation_wer,
                "lr": learning_rates,
                "grad_norm": grad_norms,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "configs": option_dict,
                "token_to_id":train_data_loader.dataset.token_to_id,
                "id_to_token":train_data_loader.dataset.id_to_token
            },
            prefix=options.prefix,
        )

        # Summary
        elapsed_time = time.time() - start_time
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        if epoch % options.print_epochs == 0 or epoch == options.num_epochs - 1:
            output_string = (
                "{epoch_text}: "
                "Train Symbol Accuracy = {train_symbol_accuracy:.5f}, "
                "Train Sentence Accuracy = {train_sentence_accuracy:.5f}, "
                "Train WER = {train_wer:.5f}, "
                "Train Loss = {train_loss:.5f}, "
                "Validation Symbol Accuracy = {validation_symbol_accuracy:.5f}, "
                "Validation Sentence Accuracy = {validation_sentence_accuracy:.5f}, "
                "Validation WER = {validation_wer:.5f}, "
                "Validation Loss = {validation_loss:.5f}, "
                "lr = {lr} "
                "(time elapsed {time})"
            ).format(
                epoch_text=epoch_text,
                train_symbol_accuracy=train_epoch_symbol_accuracy,
                train_sentence_accuracy=train_epoch_sentence_accuracy,
                train_wer=train_epoch_wer,
                train_loss=train_result["loss"],
                validation_symbol_accuracy=validation_epoch_symbol_accuracy,
                validation_sentence_accuracy=validation_epoch_sentence_accuracy,
                validation_wer=validation_epoch_wer,
                validation_loss=validation_result["loss"],
                lr=epoch_lr,
                time=elapsed_time,
            )
            print(output_string)
            # epoch base wandb log 
            wandb.log(
                {
                    "epoch": start_epoch + epoch + 1,
                    "train/loss": train_result["loss"],
                    "train/symbol_accuracy": train_epoch_symbol_accuracy,
                    "train/sentence_accuracy": train_epoch_sentence_accuracy,
                    "train/wer": train_epoch_wer,
                    "val/losses": validation_result["loss"],
                    "val/symbol_accuracy": validation_epoch_symbol_accuracy,
                    "val/sentence_accuracy":validation_epoch_sentence_accuracy,
                    "val/wer":validation_epoch_wer,
                    "val/score" : (validation_epoch_sentence_accuracy * 0.9) + (0.1 * (1 - validation_epoch_wer)),
                    # "train/teacher_forcing" : options.teacher_forcing_ratio[epoch // options.ratio_cycle]
                }
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_file",
        dest="config_file",
        default="configs/Attention.yaml",
        type=str,
        help="Path of configuration file",
    )
    parser.add_argument(
        '-w', 
        '--wandb_project', 
        type=str, 
        default='OCR',
        help='W&B project'
    )
    parser = parser.parse_args()

    # configuration
    hparams = EasyDict()
    with open(parser.config_file,'r') as f:
        hparams.update(yaml.safe_load(f))

    # wandb init
    wandb.init(project=parser.wandb_project, config=hparams)
    
    main(parser.config_file)
