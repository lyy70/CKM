from typing import Optional, Callable, Any
from torch.utils.data import Dataset
import json
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from argparse import Namespace

def generate_class_info(args: Namespace):
    class_to_idx = {}
    # obj_list = [ "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    #              "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",]
    # obj_list = [ "screw"]
    obj_list = args.item_list
    for k, index in zip(obj_list, range(len(obj_list))):
        class_to_idx[k] = index

    return obj_list, class_to_idx

class MVTec(Dataset):
    def __init__(self, root, transform_img, transform_mask, mode, args):
        self.root = root
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.data_all = []
        self.label = []
        self.data = []
        self.cls_idx = []
        self.cls_name = []
        self.img_path = []
        self.img_mask = []
        self.mode = mode
        self.obj_list, self.class_to_idx = generate_class_info(args)

        meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
        meta_info = meta_info[mode]

        self.cls_names = list(meta_info.keys())
        for cls_name in self.cls_names:
            self.data_all.extend(meta_info[cls_name])

        for data in tqdm(self.data_all, '| collection information | %s |' % self.mode):
            img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
                                                                  data['specie_name'], data['anomaly']
            self.img_path.append(img_path)
            self.label.append(anomaly)
            self.cls_name.append(cls_name)

            img = Image.open(os.path.join(self.root, img_path)).convert('RGB')
            if anomaly == 0:
                img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
            else:
                if os.path.isdir(os.path.join(self.root, mask_path)):
                    img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
                else:
                    img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
                    img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')

            self.data.append(img)
            self.img_mask.append(img_mask)
            self.cls_idx.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        cls_idx = self.cls_idx[index]
        cls_name = self.cls_name[index]
        label =self.label[index]
        img_path = self.img_paths[index]
        img_mask = self.img_mask[index]

        return img, cls_idx, cls_name, label, img_path, img_mask
