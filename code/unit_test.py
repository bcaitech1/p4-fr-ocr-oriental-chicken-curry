import os
import cv2
import random
import torch
import torch.nn as nn
import albumentations as A

from prettyprinter import pprint
from torchsummary import summary
# from torchinfo import summary
from torch.utils.data import DataLoader
from albumentations.pytorch.transforms import ToTensorV2

from dataset import LoadDataset, split_gt, collate_batch
from networks.backbone import ResNet_ASTER, TimmModel
from tia import Opening, Closing, TIADistortion, TIAStretch, TIAPerspective


save_path = '/opt/ml/code/aug_test'

groundtruth = '/opt/ml/input/data/train_dataset/gt.txt'
tokens_file = ["/opt/ml/input/data/train_dataset/tokens.txt"]
source_file = '/opt/ml/input/data/train_dataset/source.txt'
crop=True
rgb=1

transformed = A.Compose([
        # Closing(p=1),
        # Opening(p=1),        
        A.Resize(128, 486),        
        # TIADistortion(p=1),
        ToTensorV2()
    ])


train_, valid_ = split_gt(groundtruth, proportion=1.0, test_percent=0.2)
print('\n', len(valid_), type(valid_))
pprint(valid_[0])
print('\n\n')
dataset = LoadDataset(split_gt(groundtruth), tokens_file, crop, transformed, rgb)
n = random.randint(0, 100000)
sample = dataset.__getitem__(1)

try : 
    cv2.imwrite(os.path.join(save_path, f'{1}TIAStretch.jpg'), sample['image'].squeeze(0).detach().cpu().numpy())    
    print('success save')
except:
    print('save failed')
    print(type(sample['image'].detach().cpu().numpy()), sample['image'].detach().cpu().numpy().shape)

pprint(sample['path'])
pprint(sample['truth'])
pprint(sample['source'])

data_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_batch,
    )

for d in data_loader:
    print(d["image"].shape)
    break

# org_net = DeepCNN300(1, 48)
# org_net.to(device)
# summary(org_net, input_size=(1, 128, 128))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

custom_net = TimmModel()
custom_net.to(device)
summary(custom_net, input_size=(1, 128, 448))

class AddCoordinates(nn.Module):
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        batch_size, _, y_dim, x_dim = input_tensor.size()

        xx_ones = torch.ones([1, 1, 1, x_dim], dtype=torch.int32)
        yy_ones = torch.ones([1, 1, 1, y_dim], dtype=torch.int32)

        xx_range = torch.arange(y_dim, dtype=torch.int32)
        yy_range = torch.arange(x_dim, dtype=torch.int32)
        xx_range = xx_range[None, None, :, None]
        yy_range = yy_range[None, None, :, None]

        xx_channel = torch.matmul(xx_range, xx_ones)
        yy_channel = torch.matmul(yy_range, yy_ones)

        # transpose y
        yy_channel = yy_channel.permute(0, 1, 3, 2)

        xx_channel = xx_channel.float() / (y_dim - 1)
        yy_channel = yy_channel.float() / (x_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1)

        if torch.cuda.is_available:
            input_tensor = input_tensor.cuda()
            xx_channel = xx_channel.cuda()
            yy_channel = yy_channel.cuda()
        ret = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret, xx_channel, yy_channel


coord_test = AddCoordinates()

org, x, y = coord_test(torch.randn(1, 1, 7, 7))

pprint(org.shape)
print('\n')
pprint(x)
print('\n')
pprint(y)