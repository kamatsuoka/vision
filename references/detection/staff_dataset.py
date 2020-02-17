import json
import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image


class StaffDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.annotations = list(sorted(os.listdir(os.path.join(root, "annotations"))))
        self.image_sizes = {}

    def __getitem__(self, idx):
        # load images ad annotations
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        ann_path = os.path.join(self.root, "annotations", self.annotations[idx])
        img = Image.open(img_path).convert("RGB")

        with open(ann_path, 'r') as f:
            annot_d = json.load(f, encoding='utf-8')

        annotations = annot_d['annotations']
        image_size = annot_d['image_size']

        if not self.image_sizes.get(idx):
            self.image_sizes[idx] = image_size

        num_objs = len(annotations)
        boxes = []
        for annot in annotations:
            xmin = annot['left']
            ymin = annot['top']
            width = annot['width']
            height = annot['height']
            xmax = xmin + width
            ymax = ymin + height
            box = [xmin, ymin, xmax, ymax]
            boxes.append(box)

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        boxes = torch.as_tensor(boxes, dtype=torch.int16)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        bad_boxes = (boxes[:, 3] <= boxes[:, 1]) | (boxes[:, 2] <= boxes[:, 0])
        for bad_box in boxes[bad_boxes]:
            print(f'BAD: {bad_box} in {self.imgs[idx]}')

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]
        area = area[keep]
        iscrowd = iscrowd[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def get_height_and_width(self, idx):
        image_size = self.image_sizes.get(idx)
        if not image_size:
            ann_path = os.path.join(self.root, "annotations", self.annotations[idx])
            with open(ann_path, 'r') as f:
                image_size = json.load(f, encoding='utf-8')['image_size']
        return [image_size['height'], image_size['width']]
