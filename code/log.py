import os
import logging

import torch
# from torchvision import transforms

from data.augmentation import *
from data.dataset import dataset_loader

from psutil import virtual_memory


def __get_logger():
    """ 로깅을 위한 인스턴스 반환

    Returns
        logging : 로깅을 위한 객채 반환
    """

    __logger = logging.getLogger('logger')

    # # 로그 포멧 정의
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # 스트림 핸들러 정의
    stream_handler = logging.StreamHandler()
    # 각 핸들러에 포멧 지정
    stream_handler.setFormatter(formatter)
    # 로거 인스턴스에 핸들러 삽입
    __logger.addHandler(stream_handler)
    # 로그 레벨 정의
    __logger.setLevel(logging.DEBUG)

    return __logger

# 로깅을 위한 객체
logger = __get_logger()

def get_device():
    """현재 사용가능한 device를 반환

    Retruns:
        torch.device : 현재 사용 가능한 디바이스
    """
    logger.info("--------------------------------------------------")
    if torch.cuda.is_available():
        _device = torch.device("cuda")
        logger.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
        logger.info(f'We will use the GPU:{torch.cuda.get_device_name(0)}')
    else:
        _device = torch.device("cpu")
        logger.info('No GPU available, using the CPU instead.')
    logger.info("--------------------------------------------------")

    return _device

def get_enviroments_log():
    """현재 시스템에 대한 환경 로그 출력
    """
    num_gpus = torch.cuda.device_count()
    num_cpus = os.cpu_count()
    mem_size = virtual_memory().available // (1024 ** 3)
    logger.info("[+] System environments")
    logger.info("The number of gpus : {}".format(num_gpus))
    logger.info("The number of cpus : {}".format(num_cpus))
    logger.info("Memory Size : {}G".format(mem_size))
    logger.info("--------------------------------------------------\n")

def checkpoint_model_log(model_checkpoint, checkpoint):
    """체크포인트 모델을 현재 사용한다면 해당 모델에 대한 정보 로그 출력

    Args:
        model_checkpoint(dict) : checkpoint 모델에서 모델에 대한 정보
        checkpoint(dict) : checkpoint 모델에 대한 정보
    """
    if model_checkpoint:
        logger.info("[+] Checkpoint")
        logger.info("Resuming from epoch : {}".format(checkpoint["epoch"]))
        logger.info("Train Symbol Accuracy : {:.5f}".format(checkpoint["train_symbol_accuracy"][-1]))
        logger.info("Train Sentence Accuracy : {:.5f}".format(checkpoint["train_sentence_accuracy"][-1]))
        logger.info("Train WER : {:.5f}".format(checkpoint["train_wer"][-1]))
        logger.info("Train Loss : {:.5f}".format(checkpoint["train_losses"][-1]))
        logger.info("Validation Symbol Accuracy : {:.5f}".format(checkpoint["validation_symbol_accuracy"][-1]))
        logger.info("Validation Sentence Accuracy : {:.5f}".format(checkpoint["validation_sentence_accuracy"][-1]))
        logger.info("Validation WER : {:.5f}".format(checkpoint["validation_wer"][-1]))
        logger.info("Validation Loss : {:.5f}".format(checkpoint["validation_losses"][-1]))
    else:
        logger.info("initial model")
    
    logger.info("--------------------------------------------------\n")

def get_dataset(options):
    """데이터셋을 받아오는 함수

    Args:
        options(collections.namedtuple) : configuration정보

    Returns:
        Datalodaer : train data loader
        Dataloader : valid data loader
        Dataset : train data set
        Dataset : valid data set
    """
    
    train_transformed, valid_transformed, _ = get_transforms(options.augmentation, options.input_size.height,options.input_size.width)

    # train_transformed = transforms.Compose(
    #     [
    #         # Resize so all images have the same size
    #         transforms.Resize((options.input_size.height, options.input_size.width)),
    #         transforms.ToTensor(),
    #     ]
    # )

    # valid_transformed = transforms.Compose(
    #     [
    #         # Resize so all images have the same size
    #         transforms.Resize((options.input_size.height, options.input_size.width)),
    #         transforms.ToTensor(),
    #     ]
    # )

    train_data_loader, validation_data_loader, train_dataset, valid_dataset = dataset_loader(options, train_transformed, valid_transformed)

    logger.info("--------------------------------------------------")
    logger.info("[+] Data")
    logger.info("The number of train samples : {}".format(len(train_dataset)))
    logger.info("The number of validation samples : {}".format(len(valid_dataset)))
    logger.info("The number of classes : {}".format(len(train_dataset.token_to_id)))
    logger.info("--------------------------------------------------\n")

    return train_data_loader, validation_data_loader, train_dataset, valid_dataset

def opt_param_log(options,enc_params_to_optimise, dec_params_to_optimise):
    """ optimizing할 파라미터 로그

    Args:
        options(collections.namedtuple) : configuration정보
        enc_params_to_optimize(list) : 모델 인코더 파라미터
        dec_params_to_optimise(list) : 모델 디코더 파라미터
    """

    logger.info("--------------------------------------------------")
    logger.info("[+] Network")
    logger.info("Type: {}".format(options.network))
    logger.info("Encoder parameters: {}".format(sum(p.numel() for p in enc_params_to_optimise)))
    logger.info("Decoder parameters: {}".format(sum(p.numel() for p in dec_params_to_optimise)))
    logger.info("--------------------------------------------------\n")