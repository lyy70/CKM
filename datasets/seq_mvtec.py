import os
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.utils.mvtec import MVTec
import torchvision.transforms as transforms
import torchvision.models as models
from utils.conf import base_mvtec_path
from PIL import Image
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, getfeature_loader, store_masked_loaders
from datasets.utils.continual_dataset import get_previous_train_loader
from typing import Tuple
from argparse import Namespace
from datasets.transforms.denormalization import DeNormalize

class MyMVTec(MVTec):
    def __init__(self, root, transform_img, transform_mask, mode, args):
        self.mode = mode
        self.attributes = ['data', 'cls_idx', 'img_mask', 'label', 'img_path']
        self.trans = [transform_img, transform_mask]
        super(MyMVTec, self).__init__(root, transform_img, transform_mask, mode, args)

    def __getitem__(self, index: int) -> Tuple[type(Image), int]:
        ret_tuple = ()
        if self.mode == 'train':
            for i, att in enumerate(self.attributes):
                att_data = getattr(self, att)[index]
                if att == 'data':
                    #att_data = Image.fromarray(att_data, mode='RGB')
                    transform = self.trans[0]
                    att_data = transform(att_data)
                if att == 'img_mask':
                    transform = self.trans[1]
                    att_data = transform(att_data)
                ret_tuple += (att_data,)

        elif self.mode == 'test':
            for i, att in enumerate(self.attributes):
                att_data = getattr(self, att)[index]
                if att == 'data':
                    #att_data = Image.fromarray(att_data, mode='RGB')
                    transform = self.trans[0]
                    att_data = transform(att_data)
                if att == 'img_mask':
                    transform = self.trans[1]
                    att_data = transform(att_data)
                ret_tuple += (att_data, )
        return ret_tuple


class SequentialMVTec(ContinualDataset):
    NAME = 'seq-mvtec'
    SETTING = 'class-il'

    def __init__(self, args: Namespace) -> None:

        super(SequentialMVTec, self).__init__(args)

        self.normalization_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_img = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            self.normalization_transform])
        self.transform_mask = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),])

        self.train_transform = None
        self.test_transform = None

        self.train_dataset = MyMVTec(base_mvtec_path(), transform_img=self.transform_img, transform_mask = self.transform_mask, mode='train', args=args)
        self.test_dataset = MyMVTec(base_mvtec_path(), transform_img=self.transform_img, transform_mask=self.transform_mask, mode='test', args=args)

    def get_data_loaders(self):
        train_loader = getfeature_loader(self.train_dataset, self.test_dataset, setting=self)
        return train_loader
