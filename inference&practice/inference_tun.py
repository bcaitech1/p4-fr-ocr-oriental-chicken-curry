import torch
import os
# from train import id_to_string
from utils.metrics import word_error_rate, sentence_acc
from utils.checkpoint import load_checkpoint
# from torchvision import transforms
from data.dataset import LoadEvalDataset, collate_eval_batch, START, PAD
from utils.flags import Flags
from utils.utils import get_network, get_optimizer
import csv
from torch.utils.data import DataLoader
import argparse
import random
from tqdm import tqdm
import torch.nn.functional as F

from data.augmentation import *
from log import *

import sys

csv.field_size_limit(sys.maxsize)

def id_to_string(tokens, data_loader,do_eval=0):
    """token id 를 문자열로 변환하는 로직

    Args:
        tokens(list) : 토큰 아이디
        data_loader(Dataloaer) : 현재 사용하고 있는 데이터 로더
        do_eval(int): 0 - train, 이 외 - eval
    """
    result = []
    if do_eval:
        eos_id = data_loader.dataset.token_to_id["<EOS>"]
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
                elif token == eos_id:
                    break
        else:
            for token in example:
                token = token.item()
                if token != -1:
                    string += data_loader.dataset.id_to_token[token] + " "

        result.append(string)
    return result

# def id_to_string(tokens, data_loader,do_eval=0):
#     result = []
#     if do_eval:
#         special_ids = [data_loader.dataset.token_to_id["<PAD>"], data_loader.dataset.token_to_id["<SOS>"],
#                        data_loader.dataset.token_to_id["<EOS>"]]

#     for example in tokens:
#         string = ""
#         if do_eval:
#             for token in example:
#                 token = token.item()
#                 if token not in special_ids:
#                     if token != -1:
#                         string += data_loader.dataset.id_to_token[token] + " "
#         else:
#             for token in example:
#                 token = token.item()
#                 if token != -1:
#                     string += data_loader.dataset.id_to_token[token] + " "

#         result.append(string)
#     return result

# def id_to_string(tokens, data_loader,do_eval=0):
#     """token id 를 문자열로 변환하는 로직

#     Args:
#         tokens(list) : 토큰 아이디
#         data_loader(Dataloaer) : 현재 사용하고 있는 데이터 로더
#         do_eval(int): 0 - train, 이 외 - eval
#     """
#     result = []
#     if do_eval:
#         eos_id = data_loader.dataset.token_to_id["<EOS>"]
#         special_ids = [data_loader.dataset.token_to_id["<PAD>"], data_loader.dataset.token_to_id["<SOS>"],
#                        data_loader.dataset.token_to_id["<EOS>"]]

#     for example in tokens:
#         string = ""
#         if do_eval:
#             for token in example:
#                 token = token.item()
#                 if token not in special_ids:
#                     if token != -1:
#                         string += data_loader.dataset.id_to_token[token] + " "
#                 elif token == eos_id:
#                     break
#         else:
#             for token in example:
#                 token = token.item()
#                 if token != -1:
#                     string += data_loader.dataset.id_to_token[token] + " "

#         result.append(string)
#     return result


def main(parser):
    is_cuda = torch.cuda.is_available()
    checkpoint = load_checkpoint(parser.checkpoint, cuda=is_cuda)
    options = Flags(checkpoint["configs"]).get()
    torch.manual_seed(options.seed)
    random.seed(options.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    hardware = "cuda" if is_cuda else "cpu"
    device = torch.device(hardware)
    print("--------------------------------")
    print("Running {} on device {}\n".format(options.network, device))

    model_checkpoint = checkpoint["model"]
    if model_checkpoint:
        print(
            "[+] Checkpoint\n",
            "Resuming from epoch : {}\n".format(checkpoint["epoch"]),
        )
    print(options.input_size.height)

    # Augmentation
    # _, _, _, _, test_transformed = get_dataset(options)
    _, _, test_transformed = get_transforms(options.augmentation,options.input_size.height, options.input_size.width)


    dummy_gt = "\sin " * parser.max_sequence  # set maximum inference sequence

    root = os.path.join(os.path.dirname(parser.file_path), "images")
    with open(parser.file_path, "r") as fd:
        reader = csv.reader(fd, delimiter="\t")
        data = list(reader)
    test_data = [[os.path.join(root, x[0]), x[0], dummy_gt] for x in data]
    test_dataset = LoadEvalDataset(
        test_data, checkpoint["token_to_id"], checkpoint["id_to_token"], crop=False, transform=test_transformed,
        rgb=options.data.rgb
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=parser.batch_size,
        shuffle=False,
        num_workers=options.num_workers,
        collate_fn=collate_eval_batch,
    )

    print(
        "[+] Data\n",
        "The number of test samples : {}\n".format(len(test_dataset)),
    )

    # model = get_network(
    #     options.network,
    #     options,
    #     model_checkpoint,
    #     device,
    #     test_dataset,
    # )

    model = get_network(
        model_type = options.network,
        FLAGS= options,
        model_checkpoint = model_checkpoint,
        device = device, 
        train_dataset = test_dataset,
        train_group_dataset = test_dataset
    )
    model.eval()
    results = []

    for d in tqdm(test_data_loader):
        images = d["image"].float().to(device)
        expected = d["truth"]["encoded"].to(device)
        width = d["width"][0]
        height = d["height"][0]

        if options.network == "SATRN":
            output = model(images, expected, False, 0.0)
        elif options.network == "SATRN2":
            output = model(images, expected, False, 0.0, False)
        
        # Original
        decoded_values = output.transpose(1, 2)

        if height * 0.5 > width:
            ## cacluate probs
            probs = F.softmax(decoded_values, dim=1)    
            probs, _ = torch.max(probs, dim=-1)

            _, sequence = torch.topk(decoded_values, 1, dim=1)
            ## get promising index sentence
            sequence = sequence.squeeze(1)
            # sequence = sequence.squeeze(1)
            
            if (1 in sequence[0]) and (1 in sequence[1]) and (1 in sequence[2]):
                row, col = torch.nonzero(sequence==1, as_tuple=True)
                promising_index = np.argmax([probs[0][:col[row==0][0].item()].mean().item(), \
                                             probs[1][:col[row==1][0].item()].mean().item(), \
                                             probs[2][:col[row==2][0].item()].mean().item()])
            else:
                _, promising_index = probs.sum(dim=1).max(dim=0)

            sequence_str = id_to_string(sequence[promising_index].unsqueeze(0), test_data_loader, do_eval=1)
        else:
            _, sequence = torch.topk(decoded_values, 1, dim=1)
            sequence = sequence.squeeze(1)
            sequence_str = id_to_string(sequence[0].unsqueeze(0), test_data_loader, do_eval=1)

        for path, predicted in zip(d["file_path"], sequence_str):
            results.append((path, predicted))
    # for d in tqdm(test_data_loader):
    #     input = d["image"].to(device)
    #     expected = d["truth"]["encoded"].to(device)

    #     if options.network == "Attention":
    #         output = model(input, expected, False, 0.0, 0 , [0.5])
    #     elif options.network == "SATRN":
    #         output = model(input, expected, False, 0.0)
    #     elif options.network == "SRN":
    #         output = model(input)
    #         output = output[2]

    #     decoded_values = output.transpose(1, 2)
    #     _, sequence = torch.topk(decoded_values, 1, dim=1)
    #     sequence = sequence.squeeze(1)
    #     sequence_str = id_to_string(sequence, test_data_loader, do_eval=1)
    #     for path, predicted in zip(d["file_path"], sequence_str):
    #         results.append((path, predicted))

    os.makedirs(parser.output_dir, exist_ok=True)
    with open(os.path.join(parser.output_dir, "output.csv"), "w") as w:
        for path, predicted in results:
            w.write(path + "\t" + predicted + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        default="./log/satrn/checkpoints/0003.pth",
        type=str,
        help="Path of checkpoint file",
    )
    parser.add_argument(
        "--max_sequence",
        dest="max_sequence",
        default=230,
        type=int,
        help="maximun sequence when doing inference",
    )
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        default=1,
        type=int,
        help="batch size when doing inference",
    )

    eval_dir = os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/')
    file_path = os.path.join(eval_dir, 'eval_dataset/input.txt')
    parser.add_argument(
        "--file_path",
        dest="file_path",
        default=file_path,
        type=str,
        help="file path when doing inference",
    )

    output_dir = os.environ.get('SM_OUTPUT_DATA_DIR', 'submit')
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        default=output_dir,
        type=str,
        help="output directory",
    )

    # 추가 부분
    # parser.add_argument(
    #     '-p',
    #     '--preprocessing',
    #     type=str,
    #     default = "baseline",
    #     help= "image_preprocessing"
    # )

    parser = parser.parse_args()
    main(parser)