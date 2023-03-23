import cv2
import numpy as np
import json

# np.set_printoptions(threshold=np.inf)
import random
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

# from visualization import plot_img_and_mask,plot_one_box,show_seg_result
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from ..utils import letterbox, augment_hsv, random_perspective, xyxy2xywh, cutout
from lib.config import cfg

single_cls = True  # just detect vehicle


class KITTIDataset(Dataset):
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
        img_root = Path(cfg.DATASET.DATAROOT)
        label_root = Path(cfg.DATASET.LABELROOT)

        if is_train:
            indicator = (
                cfg.DATASET.TRAIN_SET
            )  # TODO: Make sure is training / testing in the config file
        else:
            indicator = cfg.DATASET.TEST_SET

        self.img_root = img_root / indicator / "image_2"
        self.label_root = label_root / indicator / "label_2"
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
        height, width = self.shapes

        for image_path in tqdm(list(self.img_list)):
            label_path = (
                str(image_path)
                .replace(str(self.img_root), str(self.label_root))
                .replace(f".{self.data_format}", ".txt")
            )

            lines = [line.split() for line in open(label_path).readlines()]

            gt = np.zeros((len(lines), 5))

            for idx, line in enumerate(lines):
                cls_id = 0 if single_cls else int(line[0])

                left = float(line[4])
                top = float(line[5])
                right = float(line[6])
                bottom = float(line[7])

                center_x = (left + right) / 2
                center_y = (top + bottom) / 2
                w = right - left
                h = bottom - top

                bbox = [center_x // width, center_y // height, w // width, h // height]
                gt[idx][0] = cls_id
                gt[idx][1:] = bbox

            rec = [{"image": str(image_path), "label": gt}]

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

        resized_shape = self.inputsize
        if isinstance(resized_shape, list):
            resized_shape = max(resized_shape)
        h0, w0 = img.shape[:2]  # orig hw
        r = resized_shape / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        h, w = img.shape[:2]

        (img, _, _), ratio, pad = letterbox(
            (img, img, img),  # TODO: This is stupid
            resized_shape,
            auto=True,
            scaleup=self.is_train,
        )
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        det_label = data["label"]
        labels = []

        if det_label.size > 0:
            # Normalized xywh to pixel xyxy format
            labels = det_label.copy()
            labels[:, 1] = (
                ratio[0] * w * (det_label[:, 1] - det_label[:, 3] / 2) + pad[0]
            )  # pad width
            labels[:, 2] = (
                ratio[1] * h * (det_label[:, 2] - det_label[:, 4] / 2) + pad[1]
            )  # pad height
            labels[:, 3] = (
                ratio[0] * w * (det_label[:, 1] + det_label[:, 3] / 2) + pad[0]
            )
            labels[:, 4] = (
                ratio[1] * h * (det_label[:, 2] + det_label[:, 4] / 2) + pad[1]
            )

        if len(labels):
            # convert xyxy to xywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

            # Normalize coordinates 0 - 1
            labels[:, [2, 4]] /= img.shape[0]  # height
            labels[:, [1, 3]] /= img.shape[1]  # width

        labels_out = torch.zeros((len(labels), 6))
        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(labels)

        img = np.ascontiguousarray(img)

        target = labels_out
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
        label_det = []
        for i, l_det in enumerate(label):
            l_det[:, 0] = i  # add target image index for build_targets()
            label_det.append(l_det)
        return (
            torch.stack(img, 0),
            [
                torch.cat(label_det, 0),
            ],
            paths,
            shapes,
        )


if __name__ == "__main__":
    dataset = KITTIDataset(cfg, True)
