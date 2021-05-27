from albumentations import *
from albumentations.pytorch import ToTensorV2


class ParameterError(Exception):
    def __init__(self):
        super().__init__('Not valid Augmentation parameter')

def get_transforms(aug_type: str, input_height, input_width):
    """
    Data augmentation 객체 생성
    Args:
        aug_type(str) : augmentation타입 지정
        input_height(int) : 재조정할 이미지 높이
        input_width(int) : 재조정할 이미지 넓이 
    Returns :
        list : train, validation, test데이터 셋에 대한 transform
    """

    if ayg_type == 'baseline':
        train_transform = Compose([
            Resize(img_size[0], img_size[1], p=1.0),
            ToTensorV2()
            ])
        val_transform = A.Compose([
            Resize(img_size[0], img_size[1], p=1.0),,
            ToTensorV2()
            ])
        test_transform = A.Compose([
            Resize(img_size[0], img_size[1], p=1.0),
            ToTensorV2()
            ])
    else:
        raise ParameterError

    return train_transform, val_transform, test_transform
