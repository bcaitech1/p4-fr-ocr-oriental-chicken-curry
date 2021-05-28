import torch.optim as optim

from networks.Attention import Attention
from networks.SATRN import SATRN
from madgrad import MADGRAD # madgrad optimizer 추가

#from madgrad import MadGrad

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
        optimizer = optim.Adam(params, lr=lr)
    elif optimizer == "Adadelta":
        optimizer = optim.Adadelta(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "Madgrad":
        optimizer = MADGRAD(params, lr=lr, momentum = 0.9,weight_decay=0, eps = 0.000001)
    else:
        raise NotImplementedError
    return optimizer