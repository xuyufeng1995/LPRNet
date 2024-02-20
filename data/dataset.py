import logging
import random
import time

import cv2
import numpy as np
import torch
from torch.utils.data import *
from tqdm import tqdm

from model.lprnet import CHARS

logger = logging.getLogger(__name__)
CHARS_DICT = {char: i for i, char in enumerate(CHARS)}


class LPRDataSet(Dataset):
    def __init__(self, data_set, img_size):

        self.data_set = data_set
        self.img_size = img_size

    def load_img(self, idx):
        img = self.data_set[idx]
        image = cv2.imread(img.src)

        # 缩放
        image = cv2.resize(image, self.img_size)[:, :, ::-1]

        # 归一化
        image = (image.astype('float32') - 127.5) * 0.007843

        # to tensor
        image = torch.from_numpy(image.transpose((2, 0, 1))).contiguous()

        if random.random() > .5:
            image = -image

        label = []
        for c in img.label:
            label.append(CHARS_DICT[c])

        return image, label

    def __getitem__(self, index):

        image, label = self.load_img(index)

        return image, label, len(label)

    def __len__(self):
        return len(self.data_set)

    @staticmethod
    def collate_fn(batch):
        images, labels, lengths = zip(*batch)
        labels = np.asarray(labels).flatten()

        return torch.stack(images, 0), torch.from_numpy(labels), lengths
