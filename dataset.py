
import csv
import os
import random
import torch
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import numpy as np

Transform=ToTensorV2()
def normalize(img, mean=0.6156548491665655, std=0.11802095108350473, max_pixel_value=255.0):
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img
class WarpMLS:
    def __init__(self, src, src_pts, dst_pts, dst_w, dst_h, trans_ratio=1.):
        self.src = src
        self.src_pts = src_pts
        self.dst_pts = dst_pts
        self.pt_count = len(self.dst_pts)
        self.dst_w = dst_w
        self.dst_h = dst_h
        self.trans_ratio = trans_ratio
        self.grid_size = 100
        self.rdx = np.zeros((self.dst_h, self.dst_w))
        self.rdy = np.zeros((self.dst_h, self.dst_w))

    @staticmethod
    def __bilinear_interp(x, y, v11, v12, v21, v22):
        return (v11 * (1 - y) + v12 * y) * (1 - x) + (v21 *
                                                      (1 - y) + v22 * y) * x

    def generate(self):
        self.calc_delta()
        return self.gen_img()

    def calc_delta(self):
        w = np.zeros(self.pt_count, dtype=np.float32)

        if self.pt_count < 2:
            return

        i = 0
        while 1:
            if self.dst_w <= i < self.dst_w + self.grid_size - 1:
                i = self.dst_w - 1
            elif i >= self.dst_w:
                break

            j = 0
            while 1:
                if self.dst_h <= j < self.dst_h + self.grid_size - 1:
                    j = self.dst_h - 1
                elif j >= self.dst_h:
                    break

                sw = 0
                swp = np.zeros(2, dtype=np.float32)
                swq = np.zeros(2, dtype=np.float32)
                new_pt = np.zeros(2, dtype=np.float32)
                cur_pt = np.array([i, j], dtype=np.float32)

                k = 0
                for k in range(self.pt_count):
                    if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
                        break

                    w[k] = 1. / (
                        (i - self.dst_pts[k][0]) * (i - self.dst_pts[k][0]) +
                        (j - self.dst_pts[k][1]) * (j - self.dst_pts[k][1]))

                    sw += w[k]
                    swp = swp + w[k] * np.array(self.dst_pts[k])
                    swq = swq + w[k] * np.array(self.src_pts[k])

                if k == self.pt_count - 1:
                    pstar = 1 / sw * swp
                    qstar = 1 / sw * swq

                    miu_s = 0
                    for k in range(self.pt_count):
                        if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
                            continue
                        pt_i = self.dst_pts[k] - pstar
                        miu_s += w[k] * np.sum(pt_i * pt_i)

                    cur_pt -= pstar
                    cur_pt_j = np.array([-cur_pt[1], cur_pt[0]])

                    for k in range(self.pt_count):
                        if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
                            continue

                        pt_i = self.dst_pts[k] - pstar
                        pt_j = np.array([-pt_i[1], pt_i[0]])

                        tmp_pt = np.zeros(2, dtype=np.float32)
                        tmp_pt[0] = np.sum(pt_i * cur_pt) * self.src_pts[k][0] - \
                                    np.sum(pt_j * cur_pt) * self.src_pts[k][1]
                        tmp_pt[1] = -np.sum(pt_i * cur_pt_j) * self.src_pts[k][0] + \
                                    np.sum(pt_j * cur_pt_j) * self.src_pts[k][1]
                        tmp_pt *= (w[k] / miu_s)
                        new_pt += tmp_pt

                    new_pt += qstar
                else:
                    new_pt = self.src_pts[k]

                self.rdx[j, i] = new_pt[0] - i
                self.rdy[j, i] = new_pt[1] - j

                j += self.grid_size
            i += self.grid_size

    def gen_img(self):
        src_h, src_w = self.src.shape[:2]
        dst = np.zeros_like(self.src, dtype=np.float32)

        for i in np.arange(0, self.dst_h, self.grid_size):
            for j in np.arange(0, self.dst_w, self.grid_size):
                ni = i + self.grid_size
                nj = j + self.grid_size
                w = h = self.grid_size
                if ni >= self.dst_h:
                    ni = self.dst_h - 1
                    h = ni - i + 1
                if nj >= self.dst_w:
                    nj = self.dst_w - 1
                    w = nj - j + 1

                di = np.reshape(np.arange(h), (-1, 1))
                dj = np.reshape(np.arange(w), (1, -1))
                delta_x = self.__bilinear_interp(
                    di / h, dj / w, self.rdx[i, j], self.rdx[i, nj],
                    self.rdx[ni, j], self.rdx[ni, nj])
                delta_y = self.__bilinear_interp(
                    di / h, dj / w, self.rdy[i, j], self.rdy[i, nj],
                    self.rdy[ni, j], self.rdy[ni, nj])
                nx = j + dj + delta_x * self.trans_ratio
                ny = i + di + delta_y * self.trans_ratio
                nx = np.clip(nx, 0, src_w - 1)
                ny = np.clip(ny, 0, src_h - 1)
                nxi = np.array(np.floor(nx), dtype=np.int32)
                nyi = np.array(np.floor(ny), dtype=np.int32)
                nxi1 = np.array(np.ceil(nx), dtype=np.int32)
                nyi1 = np.array(np.ceil(ny), dtype=np.int32)

                if len(self.src.shape) == 3:
                    x = np.tile(np.expand_dims(ny - nyi, axis=-1), (1, 1, 3))
                    y = np.tile(np.expand_dims(nx - nxi, axis=-1), (1, 1, 3))
                else:
                    x = ny - nyi
                    y = nx - nxi
                dst[i:i + h, j:j + w] = self.__bilinear_interp(
                    x, y, self.src[nyi, nxi], self.src[nyi, nxi1],
                    self.src[nyi1, nxi], self.src[nyi1, nxi1])

        dst = np.clip(dst, 0, 255)
        dst = np.array(dst, dtype=np.uint8)

        return dst
    


def tia_distort(src, segment=4):
    img_h, img_w = src.shape[:2]

    cut = img_w // segment
    thresh = cut // 3

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([np.random.randint(thresh), np.random.randint(thresh)])
    dst_pts.append(
        [img_w - np.random.randint(thresh), np.random.randint(thresh)])
    dst_pts.append(
        [img_w - np.random.randint(thresh), img_h - np.random.randint(thresh)])
    dst_pts.append(
        [np.random.randint(thresh), img_h - np.random.randint(thresh)])

    half_thresh = thresh * 0.5

    for cut_idx in np.arange(1, segment, 1):
        src_pts.append([cut * cut_idx, 0])
        src_pts.append([cut * cut_idx, img_h])
        dst_pts.append([
            cut * cut_idx + np.random.randint(thresh) - half_thresh,
            np.random.randint(thresh) - half_thresh
        ])
        dst_pts.append([
            cut * cut_idx + np.random.randint(thresh) - half_thresh,
            img_h + np.random.randint(thresh) - half_thresh
        ])

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()

    return dst


def tia_stretch(src, segment=4):
    img_h, img_w = src.shape[:2]

    cut = img_w // segment
    thresh = cut * 4 // 5

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([0, 0])
    dst_pts.append([img_w, 0])
    dst_pts.append([img_w, img_h])
    dst_pts.append([0, img_h])

    half_thresh = thresh * 0.5

    for cut_idx in np.arange(1, segment, 1):
        move = np.random.randint(thresh) - half_thresh
        src_pts.append([cut * cut_idx, 0])
        src_pts.append([cut * cut_idx, img_h])
        dst_pts.append([cut * cut_idx + move, 0])
        dst_pts.append([cut * cut_idx + move, img_h])

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()

    return dst


def tia_perspective(src):
    img_h, img_w = src.shape[:2]

    thresh = img_h // 2

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([0, np.random.randint(thresh)])
    dst_pts.append([img_w, np.random.randint(thresh)])
    dst_pts.append([img_w, img_h - np.random.randint(thresh)])
    dst_pts.append([0, img_h - np.random.randint(thresh)])

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()

    return dst
START = "<SOS>"
END = "<EOS>"
PAD = "<PAD>"
SPECIAL_TOKENS = [START, END, PAD]


# Rather ignorant way to encode the truth, but at least it works.
def encode_truth(truth, token_to_id):

    truth_tokens = truth.split()
    for token in truth_tokens:
        if token not in token_to_id:
            raise Exception("Truth contains unknown token")
    truth_tokens = [token_to_id[x] for x in truth_tokens]
    if '' in truth_tokens: truth_tokens.remove('')
    return truth_tokens


def load_vocab(tokens_paths):
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
    root = os.path.join(os.path.dirname(groundtruth), "images")
    source = os.path.join(os.path.dirname(groundtruth), "source.txt")
    with open(groundtruth, "r") as fd, open(source, "r") as src:
        data=[]
        source=[]   # for stratified split
        for line, line2 in zip(fd, src):
            tmp = line.strip().split("\t")
            tmp.append(int(line2.strip().split("\t")[1]))
            source.append(int(line2.strip().split("\t")[1]))
            data.append(tmp)
        # random.shuffle(data)
        dataset_len = round(len(data) * proportion)
        data = data[:dataset_len]
        data = [[os.path.join(root, x[0]), x[1], x[2]] for x in data]
        # print(len(data), len(source))
    
    if test_percent:
        X_train, y_train = train_test_split(data, test_size=test_percent, stratify=source, random_state=77)
        # test_len = round(len(data) * test_percent)
        # print(len(X_train), type(X_train), len(data[test_len:]), type(data[test_len:]))
        # return data[test_len:], data[:test_len]
        return X_train, y_train
    else:
        return data


def collate_batch(data):
    max_len = max([len(d["truth"]["encoded"]) for d in data])
    # Padding with -1, will later be replaced with the PAD token
    padded_encoded = [
        d["truth"]["encoded"] + (max_len - len(d["truth"]["encoded"])) * [-1]
        for d in data
    ]
    return {
        "path": [d["path"] for d in data],
        "source": torch.tensor([d["source"] for d in data]),
        "image": torch.stack([d["image"] for d in data], dim=0),
        "truth": {
            "text": [d["truth"]["text"] for d in data],
            "encoded": torch.tensor(padded_encoded)
        },
    }

def collate_eval_batch(data):
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

class LoadDataset(Dataset):
    """Load Dataset"""

    def __init__(
        self,
        groundtruth,
        tokens_file,
        crop=False,
        transform=None,
        rgb=3,
        mode='train'
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
        self.mode=mode
        self.crop = crop
        self.transform = transform
        self.rgb = rgb
        self.token_to_id, self.id_to_token = load_vocab(tokens_file)
        self.data = [
            {
                "path": p,
                "source": source,
                "truth": {
                    "text": truth,
                    "encoded": [
                        self.token_to_id[START],
                        *encode_truth(truth, self.token_to_id),
                        self.token_to_id[END],
                    ],
                },
            }
            for p, truth, source in groundtruth
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

        if self.crop:
            # Image needs to be inverted because the bounding box cuts off black pixels,
            # not white ones.
            bounding_box = ImageOps.invert(image).getbbox()
            image = image.crop(bounding_box)

        if self.transform:
            image_np = np.array(image)
            h,w=image_np.shape
            if h>w :
              image_np=image_np.reshape(w,h)
            alpha=np.random.rand()
            if self.mode=='train':
              if 0.0<alpha<0.2:
                image_np=tia_distort(image_np)
              elif 0.2<alpha<0.3:
                image_np=tia_stretch(image_np)
            image = self.transform(image=image_np)['image']
            image=normalize(image)
            image=Transform(image=image)['image']
        return {"path": item["path"], "truth": item["truth"], "image": image, "source": item["source"]}

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
            for p, p1, truth in groundtruth
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

        if self.crop:
            # Image needs to be inverted because the bounding box cuts off black pixels,
            # not white ones.
            bounding_box = ImageOps.invert(image).getbbox()
            image = image.crop(bounding_box)

        if self.transform:
            image = self.transform(image)

        return {"path": item["path"], "file_path":item["file_path"], "truth": item["truth"], "image": image}

def dataset_loader(options, train_transform, valid_transform):

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
        train_data, options.data.token_paths, crop=options.data.crop, transform=train_transform, rgb=options.data.rgb,mode='train'
    )
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=options.batch_size,
        shuffle=True,
        num_workers=options.num_workers,
        collate_fn=collate_batch,
    )

    valid_dataset = LoadDataset(
        valid_data, options.data.token_paths, crop=options.data.crop, transform=valid_transform, rgb=options.data.rgb,mode='valid'
    )
    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=options.batch_size,
        shuffle=False,
        num_workers=options.num_workers,
        collate_fn=collate_batch,
    )

    return train_data_loader, valid_data_loader, train_dataset, valid_dataset

