import os
import csv
import argparse
import random
import torch

from torch.utils.data import DataLoader

from tqdm import tqdm
from log import *

from utils.metrics import word_error_rate, sentence_acc
from utils.checkpoint import load_checkpoint
from utils.flags import Flags
from utils.utils import get_network, get_optimizer
from data.dataset import LoadEvalDataset,LoadGroupEvalDataset, collate_eval_batch
from data.vocab import START, PAD, load_group_vocab
from data.augmentation import *


import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class Denoising(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
    
    def apply(self, img, **params):
        return cv2.fastNlMeansDenoising(img, h=5)



def id_to_string(tokens, data_loader,score,do_eval=0):
    """token id 를 문자열로 변환하는 로직

    Args:
        tokens(list) : 토큰 아이디
        data_loader(Dataloaer) : 현재 사용하고 있는 데이터 로더
        do_eval(int): 0 - train, 이 외 - eval
    """
    result = []
    result_score = []
    if do_eval:
        eos_id = data_loader.dataset.token_to_id["<EOS>"]
        special_ids = [data_loader.dataset.token_to_id["<PAD>"], data_loader.dataset.token_to_id["<SOS>"],
                       data_loader.dataset.token_to_id["<EOS>"]]

    for i, example in enumerate(tokens):
        string = ""
        temp_score = 0
        if do_eval:
            for j, token in enumerate(example):
                token = token.item()
                temp_score += score[i][0][j]
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
        result_score.append(temp_score/ len(string))
        result.append(string)
    return result, result_score

def main(parser):
    is_cuda = torch.cuda.is_available()

    checkpoint1 = load_checkpoint(parser.checkpoint, cuda=is_cuda)
    model_checkpoint1 = checkpoint1["model"]
    checkpoint_model_log(model_checkpoint1,checkpoint1)
    options1 = Flags(checkpoint1["configs"]).get()

    checkpoint2 = load_checkpoint(parser.checkpoint2, cuda=is_cuda)
    model_checkpoint2 = checkpoint2["model"]
    checkpoint_model_log(model_checkpoint2,checkpoint2)
    options2 = Flags(checkpoint2["configs"]).get()


    # ========================================================
    torch.manual_seed(options1.seed)
    random.seed(options1.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    hardware = "cuda" if is_cuda else "cpu"
    device = torch.device(hardware)
    print("--------------------------------")
    print("Running {} on device {}\n".format(options1.network, device))
    # ========================================================

    # Augmentation1
    _, _, test_transformed1 = get_transforms(
        options1.augmentation,
        options1.input_size.height,
        options1.input_size.width
    )

    # vimSATRN 
    # Augmentation2
    test_transformed2 = A.Compose([
        A.Resize(options2.input_size.height, options2.input_size.width),
        Denoising(p=1),
        A.Normalize(mean=(0.6156), std=(0.1669)),
        ToTensorV2()
    ])

    # test 데이터 준비
    dummy_gt = "\sin " * parser.max_sequence  # set maximum inference sequence
    root = os.path.join(os.path.dirname(parser.file_path), "images")
    with open(parser.file_path, "r") as fd:
        reader = csv.reader(fd, delimiter="\t")
        data = list(reader)
    test_data = [[os.path.join(root, x[0]), x[0], dummy_gt] for x in data]

    # dataset1
    test_dataset1 = LoadEvalDataset(
        test_data, checkpoint1["token_to_id"], checkpoint1["id_to_token"], crop=False, transform=test_transformed1,
        rgb=options1.data.rgb
    )
    test_data_loader1 = DataLoader(
        test_dataset1,
        batch_size=parser.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_eval_batch,
    )  
    # dataset1 Group
    group_token_to_id, group_id_to_token = load_group_vocab(options1.data.token_paths)
    test_group_dataset1 = LoadGroupEvalDataset(
        test_data, group_token_to_id, group_id_to_token, crop=False, transform=test_transformed1,
        rgb=options1.data.rgb
    )

    # dataset2
    test_dataset2 = LoadEvalDataset(
        test_data,  checkpoint2["token_to_id"], checkpoint2["id_to_token"], crop=False, transform=test_transformed2,
        rgb=options2.data.rgb
    )
    test_data_loader2 = DataLoader(
        test_dataset2,
        batch_size=parser.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_eval_batch,
    )

    print("=========================================================================")
    print(
        "[+] Data\n",
        "The number of test samples : {}\n".format(len(test_dataset1)),
    )

    # model_checkpoint = checkpoint["model"]
    if model_checkpoint1:
        print(
            "[+] Checkpoint1\n",
            "Resuming from epoch : {}\n".format(checkpoint1["epoch"]),
        )
    print(options1.input_size.height)

    print(
        "[+] Data\n",
        "The number of test samples : {}\n".format(len(test_dataset2)),
    )

    # model_checkpoint = checkpoint["model"]
    if model_checkpoint2:
        print(
            "[+] Checkpoint2\n",
            "Resuming from epoch : {}\n".format(checkpoint2["epoch"]),
        )
    print(options2.input_size.height)
    print("=========================================================================")


    model1 = get_network(
        model_type = options1.network,
        FLAGS= options1,
        model_checkpoint = model_checkpoint1,
        device = device, 
        train_dataset = test_dataset1,
        train_group_datasets = [test_group_dataset1]
    )

    model2 = get_network(
        model_type = options2.network,
        FLAGS = options2,
        model_checkpoint = model_checkpoint2,
        device = device,
        train_dataset = test_dataset2,
        train_group_datasets = [test_group_dataset1]
    )

    # model1 inference
    model1 = model1.to(device)
    model1.eval()
    results1 = []
    for d in tqdm(test_data_loader1):
        input = d["image"].to(device)
        expected = d["truth"]["encoded"].to(device)

        if options1.network == "Attention":
            output = model1(input, expected, False, 0.0, 0 , [0.5])
        elif options1.network == "SATRN":
            output = model1(input, expected, False, 0.0)
        elif options1.network == "SATRN2":
            output = model1(input, expected, False, 0.0, 1) # group stat -> 가장 마지막꺼
        elif options1.network == "SRN":
            output = model1(input)
            output = output[2]

        decoded_values = output.transpose(1, 2)
        score, sequence = torch.topk(decoded_values, 1, dim=1)
        score = score.detach().cpu()
        sequence = sequence.squeeze(1)
        sequence_str, result_score = id_to_string(sequence, test_data_loader1,score, do_eval=1)

        for path, predicted, score in zip(d["file_path"], sequence_str, result_score):
            results1.append((path, predicted, score))
    
    torch.cuda.empty_cache()

    # model2 inference
    model2 = model2.to(device)
    model2.eval()
    results2 = []
    for d in tqdm(test_data_loader2):
        input = d["image"].to(device)
        expected = d["truth"]["encoded"].to(device)

        output = model2(input, expected, False, 0.0)
        
        decoded_values = output.transpose(1, 2)
        score, sequence = torch.topk(decoded_values, 1, dim=1)
        score = score.detach().cpu()
        sequence = sequence.squeeze(1)
        sequence_str, result_score = id_to_string(sequence, test_data_loader2,score, do_eval=1)
        for path, predicted,score in zip(d["file_path"], sequence_str, result_score):
            results2.append((path, predicted, score))

    os.makedirs(parser.output_dir, exist_ok=True)
    with open(os.path.join(parser.output_dir, "output.csv"), "w") as w:
        for i in range(len(results1)):
            path1, predicted1, score1 = results1[i]
            path2, predicted2, score2 = results2[i]

            if score1 <= score2:
                w.write(path1 + "\t" + predicted2 + "\n")
            else:
                w.write(path1 + "\t" + predicted1 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        default="./log/satrn/checkpoints/0049.pth",
        type=str,
        help="Path of checkpoint file",
    )

    parser.add_argument(
        "--checkpoint2",
        dest="checkpoint2",
        default="./log/vim_satrn/0006.pth",
        type=str,
        help="Path of checkpoint file2",
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
        default=2,
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