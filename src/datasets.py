import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import random

# dataset
from torch.utils.data import Dataset  # , DataLoader

# from utils.collate_fn import detr_collate_fn

# from glob import glob
import os
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2

# root_dir = "/workspace/traffic_light/data/detection/train/"

import os
from transformers import AutoImageProcessor
import sys


random.seed(42)


class TrafficLightDataset(Dataset):
    """
    self.img_keys : The index and image_id are different,
        so a mapping function is needed to map the index to the image_id.
    """

    def __init__(self, root_dir: str, mode: str, image_processor: str, transform=None):
        self.mode = mode
        self.img_paths = os.path.join(root_dir, f"{mode}/images")
        self.coco = COCO(os.path.join(root_dir, f"{mode}/{mode}.json"))
        self.length = len([img for img in os.listdir(self.img_paths) if ".jpg" in img])
        self.img_keys = sorted(self.coco.imgs.keys())
        self.transform = transform
        self.image_processor = AutoImageProcessor.from_pretrained(image_processor)

    def __call__(self, transform):
        self.transform = transform
        return self

    def __len__(self):
        return self.length

    def _load_image(self, id: int) -> Image.Image:
        id = str(id).zfill(8)
        img_path = os.path.join(self.img_paths, f"{id}.jpg")
        image = Image.open(img_path)  # .convert("RGB")
        image = np.array(image)  # H, W, C
        return image

    def _load_bbox(self, id, orig_width, orig_height):
        bboxes, labels, area = [], [], []

        for anno_id in self.coco.getAnnIds(id):
            anns = self.coco.loadAnns(anno_id)[0]
            labels.append(anns["category_id"])
            # bboxes.append(anns['bbox'])
            bboxes.append(
                # self._coco_to_pascal_bbox(id, anns["bbox"], orig_width, orig_height)
                self._coco_check_bbox(anns["bbox"], orig_width, orig_height)
            )
            area.append(anns["area"])
        return bboxes, labels, area

    def _coco_to_pascal_bbox(self, anns, orig_width, orig_height):
        x_min = anns[0]
        x_max = anns[0] + anns[2]
        y_min = anns[1]
        y_max = anns[1] + anns[3]
        if x_min < 0:
            x_min = float(0)
        if y_min < 0:
            y_min = float(0)
        if x_max > orig_width:
            x_max = float(orig_width)
        if y_max > orig_height:
            y_max = float(orig_height)
        return [x_min, y_min, x_max, y_max]

    def _coco_check_bbox(self, anns, orig_width, orig_height):
        x_min = anns[0]
        x_max = anns[0] + anns[2]
        y_min = anns[1]
        y_max = anns[1] + anns[3]
        w = anns[2]
        h = anns[3]
        if x_min < 0:
            x_min = float(0)
        if y_min < 0:
            y_min = float(0)
        if x_max > orig_width:
            w = float(orig_width - x_min)
        if y_max > orig_height:
            h = float(orig_height - y_min)
        return [x_min, y_min, w, h]

    def formatted_anns(self, id, labels, area, bboxes, orig_w, orig_h):
        """_summary_

        Args:
            id (_type_): _description_
            labels (_type_): _description_
            area (_type_): _description_
            bboxes (List[Tuple[float]]): list of bounding boxes provided in COCO format
            orig_w (_type_): _description_
            orig_h (_type_): _description_

        Returns:
            List[Dict]
        """
        annotations = []
        for i in range(0, len(labels)):
            bbox = [round(value, 3) for value in bboxes[i]]
            new_ann = {
                "image_id": id,
                "category_id": labels[i],
                "isCrowd": 0,
                "area": area,
                "bbox": bbox,
                "orig_size": [orig_w, orig_h],
            }
            annotations.append(new_ann)

        return annotations

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            image = torch.Tensor[C, H, W]
            target = {
                    'pixel_mask' : torch.Tensor[H, W]
                    'labels' : [{'size': torch.int64
                                'image_id': torch.int64
                                'class_labels': torch.int64
                                "boxes": torch.float32 -> [cx, cy, w, h]: [0, 1]
                                'area': torch.float32
                                'iscrowd': torch.int64
                                'orig_size': torch.int64 -> [W, H]
        """
        index = self.img_keys[index]

        orig_width = self.coco.loadImgs(index)[0]["width"]
        orig_height = self.coco.loadImgs(index)[0]["height"]
        image = self._load_image(index)  # type of image is np.array
        bboxes, labels, area = self._load_bbox(
            index, orig_width, orig_height
        )  # type of these are python

        # data cleansing
        if index == 26573:
            del bboxes[2]
            del labels[2]

        if self.transform is not None:
            transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
            image, bboxes = transformed["image"], transformed["bboxes"]

        annotations = {
            "image_id": index,
            "annotations": self.formatted_anns(
                index, labels, area, bboxes, orig_width, orig_height
            ),
        }
        encoding = self.image_processor(
            images=image, annotations=annotations, return_tensors="pt"
        )

        image = encoding["pixel_values"].squeeze()
        mask = encoding["pixel_mask"].squeeze()
        labels = encoding["labels"][0]  # remove batch dimension
        target = {"pixel_mask": mask, "labels": labels}

        return image, target
