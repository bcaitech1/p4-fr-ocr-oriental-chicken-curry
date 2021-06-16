import torch
import os
#from train import id_to_string
from torchvision import transforms
from data.dataset import LoadEvalDataset, collate_eval_batch, START, PAD
import csv
from torch.utils.data import DataLoader
import argparse
import random
from tqdm import tqdm


from psutil import virtual_memory

from utils.flags import Flags
from utils.utils import get_network, get_optimizer
from utils.metrics import word_error_rate, sentence_acc
from utils.scheduler import CircularLRBeta, CosineAnnealingWarmUpRestarts
from utils.checkpoint import (
    default_checkpoint,
    load_checkpoint,
    save_checkpoint,
    init_tensorboard,
    write_tensorboard,
)
import numpy as np
from data.dataset import dataset_loader, START, PAD, load_vocab
from data.augmentation import *
from log import *
import torchvision
from torchvision import transforms

from efficientnet_pytorch import EfficientNet

from albumentations import *
import albumentations as A
from albumentations.pytorch import ToTensorV2


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


def id_to_string(tokens, data_loader, do_eval=0):
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


def id_to_string_2(tokens, data_loader, do_eval=0):
    result = []
    if do_eval:
        eos_id = data_loader.dataset.token_to_id["<EOS>"]
        special_ids = [data_loader.dataset.token_to_id["<PAD>"], data_loader.dataset.token_to_id["<SOS>"],
                       data_loader.dataset.token_to_id["<EOS>"]]
    # 224 : { ,  213 :}
    for example in tokens:
        string = ""
        cou = 0  # 추가
        st = 0  # 추가
        if do_eval:
            for i, token in enumerate(example):
                token = token.item()
                # { } 개수 맞추기
                # ------ 추가 -------
                if token == 224:
                    cou += 1
                    st += 1
                    if st > 2:
                        st = 2
                        cou -= 1
                elif token == 213:
                    cou -= 1
                elif st > 0:
                    for i in range(st):
                        string += "{" + " "

                    st = 0

                if cou == -1:
                    cou = 0
                    continue

                if token == 213 and st > 0:
                    # { } _ 고려하기
                    if i + 1 < len(example) and (example[i+1] == data_loader.dataset.token_to_id["_"] or example[i+1] == data_loader.dataset.token_to_id["^"]):
                        string += "{ } "
                    st -= 1
                    continue
                # ------ 추가 -------

                # stack에 {를 저장하고 있다가 }가 아닌 입력이 들어오면 string에 넣어준다.
                # }가 들어왔는데 stack에 { 가 있다면 { pop 해주면 된다.
                if token not in special_ids:
                    if token != -1 and token != 224:      # ----추가 필수 ---------------
                        string += data_loader.dataset.id_to_token[token] + " "
                elif token == eos_id:
                    break
        else:
            for token in example:
                token = token.item()
                if token != -1:
                    string += data_loader.dataset.id_to_token[token] + " "
        # for i in range(cou):
        #     string += "}" + " "
        result.append(string)
    return result


def main(parser):

    #options = Flags(parser.config_file).get()

    is_cuda = torch.cuda.is_available()
    checkpoint = load_checkpoint(parser.checkpoint, cuda=is_cuda)
    options = Flags(checkpoint["configs"]).get()

    # set random seed
    seed_everything(options.seed)
    print("optimizer : ", options.optimizer.optimizer)
    # hardware = "cuda" if is_cuda else "cpu"
    # device = torch.device(hardware)
    # print("--------------------------------")
    # print("Running {} on device {}\n".format(options.network, device))

    device = get_device()

    model_checkpoint = checkpoint["model"]
    if model_checkpoint:
        print(
            "[+] Checkpoint\n",
            "Resuming from epoch : {}\n".format(checkpoint["epoch"]),
        )
    print(options.input_size.height)

    transformed = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((options.input_size.height,
                                           options.input_size.width)),
            torchvision.transforms.ToTensor(),
        ]
    )

    transformed_rotate = A.Compose(
        [
            A.Resize(128,
                     448, p=1.0),
            A.Normalize(
                mean=(0.6162933558268724),
                std=(0.16278683017346854),
                max_pixel_value=255.0,
                p=1.0),
            ToTensorV2(),
        ]
    )

    dummy_gt = "\sin " * parser.max_sequence  # set maximum inference sequence

    root = os.path.join(os.path.dirname(parser.file_path), "images")
    with open(parser.file_path, "r") as fd:
        reader = csv.reader(fd, delimiter="\t")
        data = list(reader)
    test_data = [[os.path.join(root, x[0]), x[0], dummy_gt] for x in data]
    test_dataset = LoadEvalDataset(
        test_data, checkpoint["token_to_id"], checkpoint["id_to_token"], crop=False, transform=transformed, transform_rotate=transformed_rotate,
        rgb=options.data.rgb
    )

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=parser.batch_size,  # parser.batch_size,
        shuffle=False,
        num_workers=options.num_workers,
        collate_fn=collate_eval_batch,
    )

    print(
        "[+] Data\n",
        "The number of test samples : {}\n".format(len(test_dataset)),
        f"batch_size : {parser.batch_size}"
    )

    model = get_network(
        options.network,
        options,
        model_checkpoint,
        device,
        test_dataset,
    )
    model.eval()

    model_rotate = EfficientNet.from_pretrained(
        "efficientnet-b1", num_classes=2)
    save_path = os.path.join(
        "/opt/ml/code/results/efficientnet-b1/009_accuracy_97.71%.ckpt")
    model_rotate.load_state_dict(torch.load(save_path))
    print(f"rotate model 성공적으로 불러옴 {save_path}")
    model_rotate.to(device)

    model_rotate.eval()

    results = []
    with torch.no_grad():
        for d in tqdm(test_data_loader):
            #print("d를 보자 ")
            # print(d.keys())
            input = d["image"].to(device)
            input_l = d["long_image"].to(device)

            # t = torch.arange(2).view(2)
            # input_l[[1], :, :, :] = torch.rot90(
            #     input_l[[1], :, :, :], 2, [2, 3])

            # input[[1], :, :, :] = torch.rot90(
            #     input[[1], :, :, :], 2, [2, 3])
            # print("----shape----")
            # print(input.shape)
            # print(input_l.shape)
            # print("------------")
            # input_l[[0], :, :, :] = torch.rot90(
            #     input_l[[0], :, :, :], 2, [2, 3])
            # print(torch.stack(
            #     [input_l, input_l, input_l], dim=1).squeeze(2).shape)

            output_r = model_rotate(torch.stack(
                [input_l, input_l, input_l], dim=1).squeeze(2))

            preds = torch.argmax(output_r, dim=-1)
            #print("rotate :", preds)
            # print("preds : ", preds)
            # print("intput_shape : ", input.shape)

            input[preds == 1, :, :, :] = torch.rot90(
                input[preds == 1, :, :, :], 2, [2, 3])

            expected = d["truth"]["encoded"].to(device)
            # print(input.shape)
            #input *= 255.0
            output = model(input, expected, False, 0.0)
            decoded_values = output.transpose(1, 2)
            _, sequence = torch.topk(decoded_values, 1, dim=1)
            sequence = sequence.squeeze(1)
            sequence_str = id_to_string(sequence, test_data_loader, do_eval=1)
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
        # default="/opt/ml/code/log/attention_50/checkpoints/0039.pth",
        default="/opt/ml/code/log/satrn/checkpoints/0070.pth",

        type=str,
        help="Path of checkpoint file",
    )
    parser.add_argument(
        "--max_sequence",
        dest="max_sequence",
        default=200,
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
