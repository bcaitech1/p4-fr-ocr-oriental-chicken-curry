import torch
import os
from math import log
import numpy as np
import torch.nn as nn
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from train import id_to_string
from utils.metrics import word_error_rate, sentence_acc
from utils.checkpoint import load_checkpoint
from torchvision import transforms
from data.dataset import LoadEvalDataset, collate_eval_batch, START, PAD
from utils.flags import Flags
from utils.utils import get_network, get_optimizer
import csv
from torch.utils.data import DataLoader
import argparse
import random
from tqdm import tqdm
import cv2

import torch.nn.functional as F

from albumentations.core.transforms_interface import ImageOnlyTransform

class Preprocessing(ImageOnlyTransform):
    
    def __init__(self,always_apply=False,p=1.0):
        super(Preprocessing, self).__init__(always_apply, p)
    
    def apply(self, img, **params):
        img = cv2.fastNlMeansDenoising(img, h = 3)

        # # Thinning and Skeletonization
        # kernel = np.ones((5,5),np.uint8)
        # img = cv2.erode(img,kernel,iterations = 1)
        return img



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

    print("inference TTA")

    # transformed = A.Compose([
    #     A.Resize(options.input_size.height, options.input_size.width),
    #     Denoising(p=1),
    #     A.Normalize(mean=(0.6156), std=(0.1669)),
    #     ToTensorV2()
    # ])

    transformed = A.Compose([
        A.Resize(options.input_size.height, options.input_size.width, p=1.0),
        Preprocessing(p = 1.0),
        A.Normalize(
            mean=(0.6162933558268724), 
            std=(0.16278683017346854), 
            p=1.0),
            ToTensorV2()
        ])

    dummy_gt = "\sin " * parser.max_sequence  # set maximum inference sequence

    root = os.path.join(os.path.dirname(parser.file_path), "images")
    with open(parser.file_path, "r") as fd:
        reader = csv.reader(fd, delimiter="\t")
        data = list(reader)
    test_data = [[os.path.join(root, x[0]), x[0], dummy_gt] for x in data]
    test_dataset = LoadEvalDataset(
        test_data, checkpoint["token_to_id"], checkpoint["id_to_token"], crop=False, transform=transformed,
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

    model = get_network(
        options.network,
        options,
        model_checkpoint,
        device,
        test_dataset,
    )
    model.eval()
    results = []
    for d in tqdm(test_data_loader):
        input = d["image"].to(device)
        expected = d["truth"]["encoded"].to(device)

        output = model(input, expected, False, 0.0)
        
        # Original
        decoded_values = output.transpose(1, 2)

        probs = F.softmax(decoded_values, dim = 1)
        # probs = decoded_values
        probs, arg_index = torch.max(probs, dim = -1)

        # print("probs shape", probs.shape)

        _, promising_index = probs.sum(dim = 1).max(dim =0)

        _, sequence = torch.topk(decoded_values, 1, dim=1)
        # sequence = sequence.squeeze(1)[promising_index].unsqueeze(0)
        sequence = sequence.squeeze(1)
        row, col = torch.nonzero(sequence==1, as_tuple=True)
        
        index = np.argmax([
            probs[0][:col[row==0][0].item()].max().item(),
            probs[1][:col[row==1][0].item()].max().item(),
            probs[2][:col[row==2][0].item()].max().item()
        ])

        # print("index", index)
        # print(probs[0][:col[row==0][0].item() + 1].mean().item() * 2.0)
        # print(probs[1][:col[row==1][0].item() + 1].mean().item())
        # print(probs[2][:col[row==2][0].item() + 1].mean().item())

        # print(probs[0][:col[row==0][0].item() + 1].mean().item() - (probs[0][:col[row==0][0].item() + 1].min().item() + probs[0][:col[row==0][0].item() + 1].max().item()) / col[row==0][0].item())
        # print(probs[1][:col[row==1][0].item() + 1].mean().item() - (probs[1][:col[row==1][0].item() + 1].min().item() + probs[1][:col[row==1][0].item() + 1].max().item()) / col[row==1][0].item())
        # print(probs[2][:col[row==2][0].item() + 1].mean().item() - (probs[2][:col[row==2][0].item() + 1].min().item() + probs[2][:col[row==2][0].item() + 1].max().item()) / col[row==2][0].item())
        # print("================================================")
        
        # print(probs[0][:col[row==0][0].item() + 1].log().neg().prod())
        # print(probs[1][:col[row==1][0].item() + 1].log().neg().prod())
        # print(probs[2][:col[row==2][0].item() + 1].log().neg().prod())
        # print("================================================")

        # print(probs[0][:col[row==0][0].item() + 1].min())
        # print(probs[1][:col[row==1][0].item() + 1].min())
        # print(probs[2][:col[row==2][0].item() + 1].min())
        # print("================================================")

        # print(probs[0][:col[row==0][0].item() + 1].max())
        # print(probs[1][:col[row==1][0].item() + 1].max())
        # print(probs[2][:col[row==2][0].item() + 1].max())
        # print("================================================")

        # print(probs[0][:col[row==0][0].item() + 1].mean())
        # print(probs[1][:col[row==1][0].item() + 1].mean())
        # print(probs[2][:col[row==2][0].item() + 1].mean())
        # print("================================================")

        # print(probs[0][:col[row==0][0].item() + 1].std())
        # print(probs[1][:col[row==1][0].item() + 1].std())
        # print(probs[2][:col[row==2][0].item() + 1].std())
        # print("================================================")

        sequence_str = id_to_string(sequence[index].unsqueeze(0), test_data_loader, do_eval=1)
        for path, predicted in zip(d["file_path"], sequence_str):
            results.append((path, predicted))

    os.makedirs(parser.output_dir, exist_ok=True)
    with open(os.path.join(parser.output_dir, "output.csv"), "w") as w:
        for path, predicted in results:
            w.write(path + "\t" + predicted + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        default="./log/satrn/checkpoints/0070.pth",
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

    parser = parser.parse_args()
    main(parser)