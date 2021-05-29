import csv
import os
import random
import torch

import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

START = "<SOS>"
END = "<EOS>"
PAD = "<PAD>"
SPECIAL_TOKENS = [START, END, PAD]


# Rather ignorant way to encode the truth, but at least it works.
def encode_truth(truth, token_to_id):
    """ ground truth의 latex문구를 파싱하여 id로 변환

    Args:
        truth(str) : gt latex
        token_to_id(dict) : token의 아이디 정보가 담겨있는 딕셔너리

    Returns:
        list : 토큰들의 아이디 정보
    """
    truth_tokens = truth.split()
    for token in truth_tokens:
        if token not in token_to_id:
            raise Exception("Truth contains unknown token")
    truth_tokens = [token_to_id[x] for x in truth_tokens]
    if '' in truth_tokens: truth_tokens.remove('')
    return truth_tokens


def load_vocab(tokens_paths):
    """ token을 id화 혹은 id로 토큰을 작을 수 있는 딕셔너리 반환

    Args:
        tokens_paths(str) : token정보가 들어있는 txt파일의 경로

    Returns:
        dict : key - token, value - id인 딕셔너리
        dict : key - id, value - token인 딕셔너리
    """
    tokens = []
    tokens.extend(SPECIAL_TOKENS)
    for tokens_file in tokens_paths:
        with open(tokens_file, "r") as fd:
            reader = fd.read()
            for token in reader.split("\n"):
                if token not in tokens:
                    tokens.append(token)
    token_to_id = {tok: i for i, tok in enumerate(tokens)}
    id_to_token = {i: tok for i, tok in enumerate(tokens)}
    return token_to_id, id_to_token


def split_gt(groundtruth, proportion=1.0, test_percent=None):
    """ground truth를 나눔

    Args:
        groundtruth(str) : ground truth의 file path
        proportion(float) : ground truth에서 얼마만큼의 비율을 가져올지
        test_percent(float) : validation data를 얼마만큼 가져갈지에 대한 비율

    Returns:
        list : train,valid data list

    """
    root = os.path.join(os.path.dirname(groundtruth), "images")
    with open(groundtruth, "r") as fd:
        data=[]
        for line in fd:
            data.append(line.strip().split("\t"))
        random.shuffle(data)
        dataset_len = round(len(data) * proportion)
        data = data[:dataset_len]
        data = [[os.path.join(root, x[0]), x[1]] for x in data]
    
    if test_percent:
        test_len = round(len(data) * test_percent)
        return data[test_len:], data[:test_len]
    else:
        return data


def collate_batch(data):
    """ train dataset에서 token id 부분에 padding을 붙히는 작업(배치 단위)

    Args:
        data(Dataset) : 데이터셋
    
    Returns:
        dict : encoded부분에 padding작업이 추가된 Dataset
    """
    # print("d[image] type", type(data[0]["image"]))
    # print("d[image] shape", data[0]["image"].shape)

    max_len = max([len(d["truth"]["encoded"]) for d in data])
    # Padding with -1, will later be replaced with the PAD token
    padded_encoded = [
        d["truth"]["encoded"] + (max_len - len(d["truth"]["encoded"])) * [-1]
        for d in data
    ]
    return {
        "path": [d["path"] for d in data],
        "image": torch.stack([d["image"] for d in data], dim=0),
        "truth": {
            "text": [d["truth"]["text"] for d in data],
            "encoded": torch.tensor(padded_encoded)
        },
    }

def collate_eval_batch(data):
    """ test dataset에서 token id 부분에 padding을 붙히는 작업(배치 단위)

    Args:
        data(Dataset) : 데이터셋
    
    Returns:
        dict : encoded부분에 padding작업이 추가된 Dataset
    """
    max_len = max([len(d["truth"]["encoded"]) for d in data])
    # Padding with -1, will later be replaced with the PAD token
    padded_encoded = [
        d["truth"]["encoded"] + (max_len - len(d["truth"]["encoded"])) * [-1]
        for d in data
    ]
    return {
        "path": [d["path"] for d in data],
        "file_path":[d["file_path"] for d in data],
        "image": torch.stack([d["image"] for d in data], dim=0),
        "truth": {
            "text": [d["truth"]["text"] for d in data],
            "encoded": torch.tensor(padded_encoded)
        },
    }

def srn_collate_batch(data):
    """ train dataset에서 token id 부분에 padding을 붙히는 작업(배치 단위)

    Args:
        data(Dataset) : 데이터셋
    
    Returns:
        dict : encoded부분에 padding작업이 추가된 Dataset
    """
    # print("d[image] type", type(data[0]["image"]))
    # print("d[image] shape", data[0]["image"].shape)

    max_len = 201
    # Padding with -1, will later be replaced with the PAD token
    padded_encoded = []
    for d in data:
        temp =  d["truth"]["encoded"] + (max_len - len(d["truth"]["encoded"])) * [-1]
        padded_encoded.append(temp[:201])

    # padded_encoded = [
    #     d["truth"]["encoded"] + (max_len - len(d["truth"]["encoded"])) * [-1]
    #     for d in data
    # ]

    return {
        "path": [d["path"] for d in data],
        "image": torch.stack([d["image"] for d in data], dim=0),
        "truth": {
            "text": [d["truth"]["text"] for d in data],
            "encoded": torch.tensor(padded_encoded)
        },
    }

def srn_collate_eval_batch(data):
    """ test dataset에서 token id 부분에 padding을 붙히는 작업(배치 단위)

    Args:
        data(Dataset) : 데이터셋
    
    Returns:
        dict : encoded부분에 padding작업이 추가된 Dataset
    """
    max_len = 201
    # Padding with -1, will later be replaced with the PAD token
    padded_encoded = []
    for d in data:
        temp =  d["truth"]["encoded"] + (max_len - len(d["truth"]["encoded"])) * [-1]
        padded_encoded.append(temp[:201])

    # padded_encoded = [
    #     d["truth"]["encoded"] + (max_len - len(d["truth"]["encoded"])) * [-1]
    #     for d in data
    # ]

    return {
        "path": [d["path"] for d in data],
        "file_path":[d["file_path"] for d in data],
        "image": torch.stack([d["image"] for d in data], dim=0),
        "truth": {
            "text": [d["truth"]["text"] for d in data],
            "encoded": torch.tensor(padded_encoded)
        },
    }

class LoadDataset(Dataset):
    """Load Dataset"""

    def __init__(
        self,
        groundtruth,
        tokens_file,
        crop=False,
        transform=None,
        rgb=3,
    ):
        """
        Args:
            groundtruth (string): Path to ground truth TXT/TSV file
            tokens_file (string): Path to tokens TXT file
            ext (string): Extension of the input files
            crop (bool, optional): Crop images to their bounding boxes [Default: False]
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(LoadDataset, self).__init__()
        self.crop = crop
        self.transform = transform
        self.rgb = rgb
        self.token_to_id, self.id_to_token = load_vocab(tokens_file)
        self.data = [
            {
                "path": p,
                "truth": {
                    "text": truth,
                    "encoded": [
                        self.token_to_id[START],
                        *encode_truth(truth, self.token_to_id),
                        self.token_to_id[END],
                    ],
                },
            }
            for p, truth in groundtruth
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        image = Image.open(item["path"])
        if self.rgb == 3:
            image = image.convert("RGB")
        elif self.rgb == 1:
            image = image.convert("L")
        else:
            raise NotImplementedError

        """세부적으로 파악해보기"""
        if self.crop:
            # Image needs to be inverted because the bounding box cuts off black pixels,
            # not white ones.
            bounding_box = ImageOps.invert(image).getbbox()
            image = image.crop(bounding_box)

        # if self.transform:
        #     image = self.transform(image)
        # print("image type", type(image))
        if self.transform:
            transformed = self.transform(image=np.array(image))
            image = transformed["image"]
            image = image.float()

        return {"path": item["path"], "truth": item["truth"], "image": image}

class LoadEvalDataset(Dataset):
    """Load Dataset"""

    def __init__(
        self,
        groundtruth,
        token_to_id,
        id_to_token,
        crop=False,
        transform=None,
        rgb=3,
    ):
        """
        Args:
            groundtruth (string): Path to ground truth TXT/TSV file
            tokens_file (string): Path to tokens TXT file
            ext (string): Extension of the input files
            crop (bool, optional): Crop images to their bounding boxes [Default: False]
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(LoadEvalDataset, self).__init__()
        self.crop = crop
        self.rgb = rgb
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.transform = transform
        self.data = [
            {
                "path": p,
                "file_path":p1,
                "truth": {
                    "text": truth,
                    "encoded": [
                        self.token_to_id[START],
                        *encode_truth(truth, self.token_to_id),
                        self.token_to_id[END],
                    ],
                },
            }
            for p, p1,truth in groundtruth
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        image = Image.open(item["path"])
        if self.rgb == 3:
            image = image.convert("RGB")
        elif self.rgb == 1:
            image = image.convert("L")
        else:
            raise NotImplementedError

        """세부적으로 파악해보기"""
        if self.crop:
            # Image needs to be inverted because the bounding box cuts off black pixels,
            # not white ones.
            bounding_box = ImageOps.invert(image).getbbox()
            image = image.crop(bounding_box)

        if self.transform:
            image = self.transform(image)
        # if self.transform:
        #     transformed = self.transform(image=np.array(image))
        #     image = transformed["image"]
        #     image = image.float()

        return {"path": item["path"], "file_path":item["file_path"],"truth": item["truth"], "image": image}

def dataset_loader(options, train_transformed, valid_transformed):
    """데이터로더 정의
    
    Args:
        options(collections.namedtuple): config정보가 담긴 객체
        train_transformed: train Augmentation transformers
        valid_transformed: valid Augmentation transformers
    """

    # Read data
    train_data, valid_data = [], [] 
    if options.data.random_split:
        for i, path in enumerate(options.data.train):
            prop = 1.0
            if len(options.data.dataset_proportions) > i:
                prop = options.data.dataset_proportions[i]
            train, valid = split_gt(path, prop, options.data.test_proportions)
            train_data += train
            valid_data += valid
    else:
        for i, path in enumerate(options.data.train):
            prop = 1.0
            if len(options.data.dataset_proportions) > i:
                prop = options.data.dataset_proportions[i]
            train_data += split_gt(path, prop)
        for i, path in enumerate(options.data.test):
            valid = split_gt(path)
            valid_data += valid

    # Load data
    train_dataset = LoadDataset(
        train_data, options.data.token_paths, crop=options.data.crop, transform=train_transformed, rgb=options.data.rgb
    )

    if options.network == "SRN":
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=options.batch_size,
            shuffle=True,
            num_workers=options.num_workers,
            collate_fn=srn_collate_batch,
        )
    else:
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=options.batch_size,
            shuffle=True,
            num_workers=options.num_workers,
            collate_fn=collate_batch,
        )

    valid_dataset = LoadDataset(
        valid_data, options.data.token_paths, crop=options.data.crop, transform=valid_transformed, rgb=options.data.rgb
    )
    if options.network == "SRN":
        valid_data_loader = DataLoader(
            valid_dataset,
            batch_size=options.batch_size,
            shuffle=False,
            num_workers=options.num_workers,
            collate_fn=srn_collate_batch,
        )
    else:
        valid_data_loader = DataLoader(
            valid_dataset,
            batch_size=options.batch_size,
            shuffle=False,
            num_workers=options.num_workers,
            collate_fn=collate_batch,
        )

    return train_data_loader, valid_data_loader, train_dataset, valid_dataset
