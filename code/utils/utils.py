import torch.optim as optim

from networks.Attention import Attention
from networks.SATRN import SATRN
from networks.SRN import SRN
from madgrad import MADGRAD # madgrad optimizer 추가


def get_network(model_type,FLAGS,model_checkpoint,device,train_dataset):
    """model설정 함수
    """
    model = None

    if model_type == "SATRN":
        model = SATRN(FLAGS, train_dataset, model_checkpoint).to(device)
    elif model_type == "CRNN":
        model = CRNN()
    elif model_type == "Attention":
        model = Attention(FLAGS, train_dataset, model_checkpoint).to(device)
    elif model_type == 'SRN':
        model = SRN(FLAGS, train_dataset).to(device)
    else:
        raise NotImplementedError

    return model


def get_optimizer(optimizer, params, lr, weight_decay=None):
    """optimizer 설정 함수
    """
    if optimizer == "Adam":
        optimizer = optim.Adam(params, lr=lr)
    elif optimizer == "Adadelta":
        optimizer = optim.Adadelta(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "Madgrad":
        optimizer = MADGRAD(params, lr = lr, momentum= 0.9, weight_decay= 0, eps= 1e-06)
    else:
        raise NotImplementedError
    return optimizer