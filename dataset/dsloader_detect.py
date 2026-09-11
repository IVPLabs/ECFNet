import os
from torch.utils.data import Dataset
from tqdm import tqdm
import random
from PIL import Image
import numpy as np
import cv2
import torch
from dataset.data_utils import *


def augment_detection(image, mask, labels):
    """Apply the paper's crop, horizontal flip and brightness augmentation."""
    h, w = mask.shape
    crop_h = np.random.randint(0, max(h // 8, 1) + 1)
    crop_w = np.random.randint(0, max(w // 8, 1) + 1)
    top = np.random.randint(0, crop_h + 1) if crop_h else 0
    left = np.random.randint(0, crop_w + 1) if crop_w else 0
    bottom = h - (crop_h - top)
    right = w - (crop_w - left)

    image = image[top:bottom, left:right]
    mask = mask[top:bottom, left:right]

    if labels.size > 0:
        boxes = labels[:, 1:5].copy()
        x1 = (boxes[:, 0] - boxes[:, 2] / 2) * w - left
        y1 = (boxes[:, 1] - boxes[:, 3] / 2) * h - top
        x2 = (boxes[:, 0] + boxes[:, 2] / 2) * w - left
        y2 = (boxes[:, 1] + boxes[:, 3] / 2) * h - top

        new_h, new_w = mask.shape
        x1, x2 = np.clip(x1, 0, new_w), np.clip(x2, 0, new_w)
        y1, y2 = np.clip(y1, 0, new_h), np.clip(y2, 0, new_h)
        keep = (x2 - x1 > 1) & (y2 - y1 > 1)
        labels = labels[keep].copy()
        x1, y1, x2, y2 = x1[keep], y1[keep], x2[keep], y2[keep]
        if len(labels):
            labels[:, 1] = ((x1 + x2) / 2) / new_w
            labels[:, 2] = ((y1 + y2) / 2) / new_h
            labels[:, 3] = (x2 - x1) / new_w
            labels[:, 4] = (y2 - y1) / new_h

    if np.random.rand() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])
        if labels.size > 0:
            labels[:, 1] = 1.0 - labels[:, 1]

    contrast = np.random.uniform(0.5, 1.5)
    brightness = np.random.randint(-20, 21)
    image = np.clip(image.astype(np.float32) * contrast + brightness, 0, 255)
    return image, mask, labels


class Data(Dataset):
    def __init__(self, root, args, folder, gt_attention_aug=False):
        super(Data, self).__init__()
        self.folder = folder
        self.root = root
        self.name_list = self.collect_data_names()
        self.gt_attention_aug = gt_attention_aug
        print('number of ' + folder + ' images:', len(self.name_list))

        self.size = [args.imgsz, args.imgsz] if isinstance(args.imgsz, int) else [x for x in args.imgsz]
        self.aug = args.aug
        self.norm = args.norm
        self.teacher_dataset_path = getattr(args, 'teacher_dataset_path', None)
        self.use_teacher = getattr(args, 'use_teacher', False)

    def collect_data_names(self):
        name_list = []
        img_folder = os.path.join(self.root, 'images/' + self.folder)
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

    def load_data(self, name):
        img_path = self._resolve_file(
            os.path.join(self.root, 'images/' + self.folder),
            name,
            ('.png', '.jpg', '.jpeg', '.bmp'),
        )
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]

        mask_path = self._resolve_file(
            os.path.join(self.root, 'masks/' + self.folder),
            name,
            ('.png', '.jpg', '.jpeg', '.bmp'),
        )
        mask = Image.open(mask_path)
        mask = np.asarray(mask)
        if len(mask.shape) == 3:
            mask = mask[..., 0]
        mask = mask / (mask.max() + 1e-8)

        teacher_image_path = None
        if self.folder == "train" and self.use_teacher:
            teacher_root = self.teacher_dataset_path or os.path.join(
                self.root, 'images', self.folder
            )
            teacher_image_path = self._resolve_file(
                teacher_root, name, ('.png', '.jpg', '.jpeg', '.bmp')
            )

        label_path = os.path.join(self.root, 'labels/' + self.folder, name + '.txt')
        if os.path.exists(label_path):
            with open(label_path) as f:
                yolo_labels = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
        else:
            yolo_labels = np.zeros((0, 5), dtype=np.float32)

        if self.norm:
            mean, std = self.compute_mean_std(img)
            img = (img - mean) / (std + 1e-8)

        img = img / (img.max() + 1e-8)

        img = cv2.resize(img, self.size, interpolation=cv2.INTER_CUBIC).astype(np.float32)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST).astype(np.float32)
        h, w = self.size[1], self.size[0]

        shapes = (
            (h0, w0),
            ((h / h0, w / w0), (0, 0))
        )

        if yolo_labels.size > 0:
            yolo_labels[:, 1:] = xywhn2xyxy(yolo_labels[:, 1:], self.size[0], self.size[1], 0, 0)
            yolo_labels[:, 1:] = xyxy2xywh(yolo_labels[:, 1:5])
            yolo_labels[:, [1, 3]] /= w
            yolo_labels[:, [2, 4]] /= h

        img = torch.from_numpy(img.transpose(2, 0, 1))
        mask = torch.from_numpy(mask).unsqueeze(0)

        labels_out = torch.zeros((len(yolo_labels), 6))
        if len(yolo_labels) > 0:
            labels_out[:, 1:] = torch.from_numpy(yolo_labels)


        if self.folder == "train":
            if self.gt_attention_aug:
                gt_attention_mask = [
                    generate_attention_mask(mask, (80, 80)),
                    generate_attention_mask(mask, (40, 40)),
                    generate_attention_mask(mask, (20, 20))
                ]
                return img, mask, labels_out, gt_attention_mask, shapes, teacher_image_path
            return img, mask, labels_out, shapes, teacher_image_path
        else:
            return img, mask, labels_out, shapes


    def compute_mean_std(self, image):
        mean = np.mean(image, axis=(0, 1))
        std = np.std(image, axis=(0, 1))
        return mean, std

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        name = self.name_list[index]
        data = self.load_data(name)

        if self.folder == 'train':
            if data is None:
                raise IndexError(f"skip...: {name}")
            return data
        else:
            return (*data, name) if isinstance(data, tuple) else (data, name)

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh
    return y

def xyxy2xywh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y

def custom_collate_fn(batch):
    is_test = len(batch[0]) == 4
    is_train_with_aug = len(batch[0]) == 6 and not is_test

    if is_test:
        imgs, masks, labels, shapes, names = zip(*batch)
        teacher_paths = None
    elif is_train_with_aug:
        imgs, masks, labels, gt_attention_masks, shapes, teacher_paths = zip(*batch)
    else:
        imgs, masks, labels, shapes, teacher_paths = zip(*batch)

    imgs = torch.stack(imgs, 0)
    masks = torch.stack(masks, 0)

    nl = [len(lb) for lb in labels]
    if sum(nl) == 0:
        labels = torch.zeros((0, 6), dtype=torch.float32)
    else:
        labels = torch.cat(labels, 0)
        offsets = torch.tensor([sum(nl[:i]) for i in range(len(nl))])
        for i in range(len(nl)):
            if nl[i] > 0:
                labels[offsets[i]:offsets[i]+nl[i], 0] = i

    if is_test:
        return imgs, masks, labels, shapes, names
    elif is_train_with_aug:
        gt_attention_masks = [torch.stack([m[i] for m in gt_attention_masks], 0) for i in range(len(gt_attention_masks[0]))]
        return imgs, masks, labels, gt_attention_masks, shapes, teacher_paths
    return imgs, masks, labels, shapes, teacher_paths





