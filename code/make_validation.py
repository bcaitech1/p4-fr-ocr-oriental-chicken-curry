import numpy as np
from pprint import pprint

from dataset import split_gt


groundtruth = '/opt/ml/input/data/train_dataset/gt.txt'
train_, valid_ = split_gt(groundtruth, proportion=1.0, test_percent=0.2)

valid_gt_path = '/opt/ml/input/data/eval_dataset/test.txt'
valid_input_path = '/opt/ml/input/data/eval_dataset/val_input.txt'

pprint(np.array(valid_).shape)

with open(valid_gt_path, mode='w', encoding='utf-8') as f:
    for line in valid_:
        f.write('D:/workspace2/input/data/train_dataset/images/' + line[0].split('/')[-1] + '\t' + line[1] + '\t' + str(line[2]) + '\n')

# with open(valid_input_path, mode='w', encoding='utf-8') as f:
#     for line in valid_:
#         f.write(line[0] + '\t' + line[1] + '\n')