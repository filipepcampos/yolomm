import cv2
import numpy as np

# np.set_printoptions(threshold=np.inf)
import random
import torch
import torchvision.transforms as transforms

# from visualization import plot_img_and_mask,plot_one_box,show_seg_result
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from ..utils import letterbox, augment_hsv, random_perspective, xyxy2xywh, cutout


class AutoDriveDataset(Dataset):
    """
    A general Dataset for some common function
    """

    def __init__(self, cfg, is_train, input_size=640, transform=None):
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
        self.config = cfg
        self.transform = transform
        self.input_size = input_size
        self.Tensor = transforms.ToTensor()

        # Dataset location
        img_root = Path(cfg.DATASET.DATAROOT)
        label_root = Path(cfg.DATASET.LABELROOT)
        mask_root = Path(cfg.DATASET.MASKROOT)
        lane_root = Path(cfg.DATASET.LANEROOT)

        indicator = cfg.DATASET.TRAIN_SET if is_train else cfg.DATASET_TEST_SET

        self.img_root = img_root / indicator
        self.label_root = label_root / indicator
        self.mask_root = mask_root / indicator
        self.lane_root = lane_root / indicator
        self.mask_list = self.mask_root.iterdir()

        self.db = []

        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.shapes = np.array(cfg.DATASET.ORG_IMG_SIZE)

    def _get_db(self):
        """
        finished on children Dataset(for dataset which is not in Bdd100k format, rewrite children Dataset)
        """
        raise NotImplementedError

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

    def __getitem__(self, index):
        """
        Get input and groud-truth from database & add data augmentation on input

        Inputs:
        -image: the index of image in self.db(database)(list)
        self.db(list) [a,b,c,...]
        a: (dictionary){'image':, 'information':}

        Returns:
        TODO
        -image: transformed image, first passed the data augmentation in __getitem__ function(type:numpy), then apply self.transform
        -target: ground truth(det_gt,seg_gt)

        function maybe useful
        cv2.imread
        cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        cv2.warpAffine
        """
        data = self.db[index]

        image = cv2.imread(  # TODO: port to pytorch
            data["image"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # seg_label = cv2.imread(data["mask"], 0)
        if self.config.num_seg_class == 3:
            seg_label = cv2.imread(data["mask"])
        else:
            seg_label = cv2.imread(data["mask"], 0)
        lane_label = cv2.imread(data["lane"], 0)

        resized_shape = self.input_size
        if isinstance(resized_shape, list):
            resized_shape = max(resized_shape)

        h0, w0 = image.shape[:2]  # orig hw
        r = resized_shape / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            image = cv2.resize(image, (int(w0 * r), int(h0 * r)), interpolation=interp)
            seg_label = cv2.resize(
                seg_label, (int(w0 * r), int(h0 * r)), interpolation=interp
            )
            lane_label = cv2.resize(
                lane_label, (int(w0 * r), int(h0 * r)), interpolation=interp
            )
        h, w = image.shape[:2]

        (image, seg_label, lane_label), ratio, pad = letterbox(
            (image, seg_label, lane_label),
            resized_shape,
            auto=True,
            scaleup=self.is_train,
        )
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
        # ratio = (w / w0, h / h0)
        # print(resized_shape)

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
            labels[:, [2, 4]] /= image.shape[0]  # height
            labels[:, [1, 3]] /= image.shape[1]  # width

        labels_out = torch.zeros((len(labels), 6))
        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(labels)
        image = np.ascontiguousarray(image)

        if self.config.num_seg_class == 3:
            _, seg0 = cv2.threshold(seg_label[:, :, 0], 128, 255, cv2.THRESH_BINARY)
            _, seg1 = cv2.threshold(seg_label[:, :, 1], 1, 255, cv2.THRESH_BINARY)
            _, seg2 = cv2.threshold(seg_label[:, :, 2], 1, 255, cv2.THRESH_BINARY)
        else:
            _, seg1 = cv2.threshold(seg_label, 1, 255, cv2.THRESH_BINARY)
            _, seg2 = cv2.threshold(seg_label, 1, 255, cv2.THRESH_BINARY_INV)
        _, lane1 = cv2.threshold(lane_label, 1, 255, cv2.THRESH_BINARY)
        _, lane2 = cv2.threshold(lane_label, 1, 255, cv2.THRESH_BINARY_INV)


        if self.config.num_seg_class == 3:
            seg0 = self.Tensor(seg0)
        seg1 = self.Tensor(seg1)
        seg2 = self.Tensor(seg2)
        lane1 = self.Tensor(lane1)
        lane2 = self.Tensor(lane2)

        if self.config.num_seg_class == 3:
            seg_label = torch.stack((seg0[0], seg1[0], seg2[0]), 0)
        else:
            seg_label = torch.stack((seg2[0], seg1[0]), 0)

        lane_label = torch.stack((lane2[0], lane1[0]), 0)

        target = [labels_out, seg_label, lane_label]
        image = self.transform(image)

        return image, target, data["image"], shapes

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
        label_det, label_seg, label_lane = [], [], []
        for i, l in enumerate(label):
            l_det, l_seg, l_lane = l
            l_det[:, 0] = i  # add target image index for build_targets()
            label_det.append(l_det)
            label_seg.append(l_seg)
            label_lane.append(l_lane)
        return (
            torch.stack(img, 0),
            [
                torch.cat(label_det, 0),
                torch.stack(label_seg, 0),
                torch.stack(label_lane, 0),
            ],
            paths,
            shapes,
        )
