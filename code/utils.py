import os
import yaml
import random
import torch
import torch.optim as optim
import numpy as np

from adamp import AdamP
from madgrad import MADGRAD
from torch.optim.adamw import AdamW
from easydict import EasyDict

from networks.Attention import Attention
from networks.SATRN import SATRN


def seed_everything(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
def get_network(
    model_type,
    FLAGS,
    model_checkpoint,
    device,
    train_dataset,
):
    model = None

    if model_type == "SATRN":
        model = SATRN(FLAGS, train_dataset, model_checkpoint).to(device)
    elif model_type == "CRNN":
        model = CRNN()
    elif model_type == "Attention":
        model = Attention(FLAGS, train_dataset, model_checkpoint).to(device)
    else:
        raise NotImplementedError

    return model


def get_optimizer(optimizer, params, lr, weight_decay=None):
    if optimizer == "Adam":
        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "AdamP":
        optimizer = AdamP(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "MADGRAD":
        optimizer = MADGRAD(params, lr=lr, weight_decay=0.0)
    elif optimizer == "AdamW":
        optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    return optimizer


class YamlConfigManager:
    def __init__(self, config_file_path='./configs/SATRN.yaml'):
        super().__init__()
        self.values = EasyDict()        
        if config_file_path:
            self.config_file_path = config_file_path
            self.reload()
    
    def reload(self):
        self.clear()
        if self.config_file_path:
            with open(self.config_file_path, 'r') as f:
                self.values.update(yaml.safe_load(f))

    def clear(self):
        self.values.clear()
    
    def update(self, yml_dict):
        for (k1, v1) in yml_dict.items():
            if isinstance(v1, dict):
                for (k2, v2) in v1.items():
                    if isinstance(v2, dict):
                        for (k3, v3) in v2.items():
                            self.values[k1][k2][k3] = v3
                    else:
                        self.values[k1][k2] = v2
            else:
                self.values[k1] = v1

    def export(self, save_file_path):
        if save_file_path:
            with open(save_file_path, 'w') as f:
                yaml.dump(dict(self.values), f)