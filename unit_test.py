import torch
import torch.nn as nn
import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2
from albumentations.pytorch.transforms import ToTensor
from torch.utils.data import DataLoader

from data.dataset import LoadDataset, split_gt, collate_batch
from prettyprinter import pprint
# from networks.backbone import ResNet_ASTER

groundtruth = '/opt/ml/input/data/train_dataset/gt.txt'
tokens_file = ["/opt/ml/input/data/train_dataset/tokens.txt"]
source_file = '/opt/ml/input/data/train_dataset/source.txt'
crop=True
rgb=1

transformed = A.Compose([
        A.Resize(128, 128),
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


train_, valid_ = split_gt(groundtruth, proportion=1.0, test_percent=0.2)
print('\n', len(valid_), type(valid_))
pprint(valid_[0])
print('\n\n')
dataset = LoadDataset(split_gt(groundtruth), tokens_file, crop, transformed, rgb)
sample = dataset.__getitem__(0)
pprint(sample['path'])
pprint(sample['truth'])
# pprint(sample['source'])

data_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_batch,
    )

for d in data_loader:
    print(d["image"].shape)
    temp = d["image"].unsqueeze(1)
    print(temp.shape)
    break