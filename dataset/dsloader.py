import os
from torch.utils.data import Dataset
from tqdm import tqdm
import random
from PIL import Image
import numpy as np
import cv2
import torch
from dataset.data_utils import *

class Data(Dataset):
    def __init__(self, root, args, folder, gt_attention_aug=False):
        super(Data, self).__init__()
        self.folder = folder
        self.root = root
        self.name_list = self.collect_data_names()
        self.gt_attention_aug = gt_attention_aug
        print('number of'+folder+' images:', len(self.name_list))

        if isinstance(args.imgsz, int):
            self.size = [args.imgsz, args.imgsz]
        else:
            self.size = [x for x in args.imgsz]
        self.aug = args.aug
        self.norm = args.norm

    # collect all image names
    def collect_data_names(self):
        name_list = []
        img_folder = os.path.join(self.root, 'images/'+self.folder)
        for img in tqdm(os.listdir(img_folder)):
            name = img.split('.')[0]
            name_list.append(name)
        random.shuffle(name_list)
        return name_list

    @staticmethod
    def _resolve_file(folder, name, extensions):
        for extension in extensions:
            path = os.path.join(folder, name + extension)
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(f"No file for '{name}' found in {folder}")

    def load_all_labels(self):
        labels = {}
        label_folder = os.path.join(self.root, 'labels/' + self.folder)

        for name in self.name_list:
            label_path = os.path.join(label_folder, name + '.txt')
            if os.path.exists(label_path):
                with open(label_path) as f:
                    lbl = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                labels[name] = lbl
            else:
                labels[name] = np.zeros((0, 5), dtype=np.float32)
        return labels

    # load data and label according to image name
    def load_data(self, name):
        img_folder = os.path.join(self.root, 'images/' + self.folder)
        label_folder = os.path.join(self.root, 'masks/' + self.folder)
        img_path = self._resolve_file(img_folder, name, ('.png', '.jpg', '.jpeg', '.bmp'))
        label_path = self._resolve_file(label_folder, name, ('.png', '.jpg', '.jpeg', '.bmp'))


        x = Image.open(img_path)
        x = np.asarray(x)
        y = Image.open(label_path)
        y = np.asarray(y)

        if len(y.shape) == 3:
            y = y[..., 0]
        y = y / (y.max() + 10e-8)

        if self.aug:
            x, y = random_crop(x, y)
            # Keep the augmentation pipeline identical to dpr-KD/run.py.
            x, y = random_rotate(x, y)
            x = random_light(x)

        if self.norm:
            mean, std = self.compute_mean_std(x)
            x = (x - mean) / (std + 10e-8)

        x = x / (x.max() + 10e-8)

        x = cv2.resize(x, self.size, interpolation=cv2.INTER_CUBIC).astype(np.float32)
        y = cv2.resize(y, self.size, interpolation=cv2.INTER_NEAREST).astype(np.float32)

        x = np.transpose(x, (2, 0, 1))
        y = y.reshape((1, self.size[0], self.size[1]))

        gt_attention_mask_80 = generate_attention_mask(y, target_size=(80, 80))
        gt_attention_mask_40 = generate_attention_mask(y, target_size=(40, 40))
        gt_attention_mask_20 = generate_attention_mask(y, target_size=(20, 20))

        gt_attention_mask = [gt_attention_mask_80, gt_attention_mask_40, gt_attention_mask_20]

        if self.folder == 'train':
            if self.gt_attention_aug:
                return x, y, gt_attention_mask

            return x, y
        else:
            return x, y, name

    def compute_mean_std(self, image):
        mean = np.zeros(3)
        std = np.zeros(3)

        for c in range(3):
            mean[c] = np.mean(image[..., c])
            std[c] = np.std(image[..., c])

        return mean, std



    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, index):
        return self.load_data(self.name_list[index])

