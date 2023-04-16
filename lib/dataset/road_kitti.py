import cv2
import numpy as np
import json

# np.set_printoptions(threshold=np.inf)
import random
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

import argparse
import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import matplotlib.pyplot as plt


# from visualization import plot_img_and_mask,plot_one_box,show_seg_result
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from ..utils import letterbox, augment_hsv, random_perspective, xyxy2xywh, cutout
from lib.config import cfg

single_cls = True  # just detect vehicle


class KITTIRoadDataset(Dataset):
    def __init__(self, cfg, is_train, inputsize=640, transform=None):
        """
        initial all the characteristic

        Inputs:
        -cfg: configurations
        -is_train(bool): whether train set or not
        -transform: ToTensor and Normalize

        Returns:
        None
        """
        self.is_train = is_train
        self.cfg = cfg
        self.transform = transform
        self.inputsize = inputsize
        self.Tensor = transforms.ToTensor()

        root = Path(cfg.DATASET.ROAD_ROOT)

        if is_train:
            indicator = (
                cfg.DATASET.TRAIN_SET
            )  # TODO: Make sure is training / testing in the config file
        else:
            indicator = cfg.DATASET.TEST_SET
        self.img_root = root / indicator / "image_2"
        self.mask_root = root / indicator / "gt_image_2"

        self.img_list = self.img_root.iterdir()

        self.db = []

        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.color_rgb = cfg.DATASET.COLOR_RGB

        # self.target_type = cfg.MODEL.TARGET_TYPE
        self.shapes = np.array(cfg.DATASET.ORG_IMG_SIZE)

        self.db = self._get_db()

    def _get_db(self):
        """
        get database from the annotation file

        Inputs:

        Returns:
        gt_db: (list)database   [a,b,c,...]
                a: (dictionary){'image':, 'information':, ......}
        image: image path
        mask: path of the segmetation label
        label: [cls_id, center_x//256, center_y//256, w//256, h//256] 256=IMAGE_SIZE
        """
        print("building database...")
        gt_db = []

        for image_path in tqdm(list(self.img_list)):
            mask_path = str(image_path).replace(str(self.img_root), str(self.mask_root))
            split_path = mask_path.split("_")
            split_path[-1] = f"road_{split_path[-1]}"
            mask_path = "_".join(split_path)

            rec = [{"image": str(image_path), "mask": mask_path}]

            gt_db += rec

        print("database build finish")
        return gt_db

    def evaluate(self, cfg, preds, output_dir):
        """
        finished on children dataset
        """
        raise NotImplementedError

    def __len__(
        self,
    ):
        """
        number of objects in the dataset
        """
        return len(self.db)

    def __getitem__(self, idx):
        """
        Get input and groud-truth from database & add data augmentation on input

        Inputs:
        -idx: the index of image in self.db(database)(list)
        self.db(list) [a,b,c,...]
        a: (dictionary){'image':, 'information':}

        Returns:
        -image: transformed image, first passed the data augmentation in __getitem__ function(type:numpy), then apply self.transform
        -target: ground truth(det_gt,seg_gt)

        function maybe useful
        cv2.imread
        cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        cv2.warpAffine
        """
        data = self.db[idx]
        img = cv2.imread(
            data["image"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        seg_label = cv2.imread(data["mask"])

        resized_shape = self.inputsize
        if isinstance(resized_shape, list):
            resized_shape = max(resized_shape)
        h0, w0 = img.shape[:2]  # orig hw
        r = resized_shape / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            seg_label = cv2.resize(
                seg_label, (int(w0 * r), int(h0 * r)), interpolation=interp
            )
        h, w = img.shape[:2]

        (img, seg_label, _), ratio, pad = letterbox(
            (img, seg_label, img),  # TODO: This is stupid
            resized_shape,
            auto=True,
            scaleup=self.is_train,
        )
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        # Convert
        img = np.ascontiguousarray(img)
        seg_label = np.ascontiguousarray(seg_label)

        _, seg0 = cv2.threshold(seg_label[:, :, 0], 1, 255, cv2.THRESH_BINARY)
        _, seg1 = cv2.threshold(seg_label[:, :, 0], 1, 255, cv2.THRESH_BINARY_INV)

        seg0 = self.Tensor(seg0)
        seg1 = self.Tensor(seg1)

        seg_label = torch.stack((seg1[0], seg0[0]), 0)

        target = seg_label
        img = self.transform(img)

        return img, target, data["image"], shapes

    def select_data(self, db):
        """
        You can use this function to filter useless images in the dataset

        Inputs:
        -db: (list)database

        Returns:
        -db_selected: (list)filtered dataset
        """
        db_selected = ...
        return db_selected

    @staticmethod
    def collate_fn(batch):
        img, label, paths, shapes = zip(*batch)
        label_seg = []
        for i, l in enumerate(label):
            label_seg.append(l)
        return (
            torch.stack(img, 0),
            [torch.stack(label_seg, 0)],
            paths,
            shapes,
        )
