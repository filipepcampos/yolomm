import cv2
import numpy as np
import json
import os

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
from ..dataset.laserscan import LaserScan, SemLaserScan
from lib.config import cfg

single_cls = True  # just detect vehicle

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']

def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)


def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)

class MultimodalKITTIDatasetLIDAR(Dataset):
    def __init__(self, cfg, is_train, labels, color_map, learning_map, learning_map_inv, sensor, max_points, inputsize=640, transform=None, sequences=None):
        self.labels = labels
        self.color_map = color_map
        self.learning_map = learning_map
        self.learning_map_inv = learning_map_inv
        self.sensor = sensor
        self.sensor_img_H = sensor["img_prop"]["height"]
        self.sensor_img_W = sensor["img_prop"]["width"]
        self.sensor_img_means = torch.tensor(sensor["img_means"], dtype=torch.float)
        self.sensor_img_stds = torch.tensor(sensor["img_stds"], dtype=torch.float)
        self.sensor_fov_up = sensor["fov_up"]
        self.sensor_fov_down = sensor["fov_down"]
        self.max_points = max_points
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

        self.img_root = img_root
        self.label_root = label_root
        self.img_list = self.img_root.iterdir()

        self.db = []

        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.color_rgb = cfg.DATASET.COLOR_RGB

        # self.target_type = cfg.MODEL.TARGET_TYPE
        self.shapes = np.array(cfg.DATASET.ORG_IMG_SIZE)

        # RangeNet parameters
        # save deats
        self.cfg = cfg

        self.sequences = sequences
        self.labels = labels
        self.color_map = color_map
        self.learning_map = learning_map
        self.learning_map_inv = learning_map_inv
        self.sensor = sensor
        self.sensor_img_H = sensor["img_prop"]["height"]
        self.sensor_img_W = sensor["img_prop"]["width"]
        self.sensor_img_means = torch.tensor(sensor["img_means"], dtype=torch.float)
        self.sensor_img_stds = torch.tensor(sensor["img_stds"], dtype=torch.float)
        self.sensor_fov_up = sensor["fov_up"]
        self.sensor_fov_down = sensor["fov_down"]
        self.max_points = max_points

        # get number of classes (can't be len(self.learning_map) because there
        # are multiple repeated entries, so the number that matters is how many
        # there are for the xentropy)
        self.nclasses = len(self.learning_map_inv)

        # sanity checks
        # make sure labels is a dict
        assert isinstance(self.labels, dict)

        # make sure color_map is a dict
        assert isinstance(self.color_map, dict)

        # make sure learning_map is a dict
        assert isinstance(self.learning_map, dict)

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

        for seq in self.sequences:
            seq = '{0:02d}'.format(int(seq))

            print("parsing seq {}".format(seq))

            # get paths for each
            sequence_scan_path = os.path.join(self.label_root, seq, "velodyne")
            sequence_label_path = os.path.join(self.label_root, seq, "labels")
            img_path = Path(os.path.join(self.img_root, seq, "image_2"))

            self.img_list = img_path.iterdir()

            for image_path in tqdm(list(self.img_list)):
                scan_path = os.path.join(sequence_scan_path, image_path.stem + ".bin")
                label_path = os.path.join(sequence_label_path, image_path.stem + ".label")

                rec = [{"image": str(image_path), "label_path": label_path, "scan_path": scan_path}]
                gt_db += rec

        print("database build finish")
        return gt_db

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

        # if r != 1:  # always resize down, only resize up if training with augmentation
        #     interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        #     img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        img = cv2.resize(img, (512, 256), interpolation=cv2.INTER_LINEAR)
        h, w = img.shape[:2]

        (img, _, _), ratio, pad = letterbox(
            (img, img, img),  # TODO: This is stupid
            resized_shape,
            auto=True,
            scaleup=self.is_train,
        )
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        labels_out = []
        lidar_data = self.get_lidar_data(idx)
        (proj,
            proj_mask,
            proj_labels,
            unproj_labels,
            path_seq,
            path_name,
            proj_x,
            proj_y,
            proj_range,
            unproj_range,
            proj_xyz,
            unproj_xyz,
            proj_remission,
            unproj_remissions,
            unproj_n_points) = lidar_data

        img = np.ascontiguousarray(img)

        target = labels_out
        img = self.transform(img)

        return img, target, data["image"], shapes, proj, proj_labels, lidar_data

    def get_bounding_boxes(self, det_label, img, ratio, pad, h, w):
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
        return labels_out
    
    def get_lidar_data(self, idx):
        # get item in tensor shape
        scan_file = self.db[idx]["scan_path"]
        label_file = self.db[idx]["label_path"]

        # open a semantic laserscan
        scan = SemLaserScan(
            self.color_map,
            project=True,
            H=self.sensor_img_H,
            W=self.sensor_img_W,
            fov_up=self.sensor_fov_up,
            fov_down=self.sensor_fov_down)
        
        scan.open_scan(scan_file)
        scan.open_label(label_file)

        # map unused classes to used classes (also for projection)
        scan.sem_label = self.map(scan.sem_label, self.learning_map)
        scan.proj_sem_label = self.map(scan.proj_sem_label, self.learning_map)

        # make a tensor of the uncompressed data (with the max num points)
        unproj_n_points = scan.points.shape[0]
        unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
        unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
        unproj_range = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
        unproj_remissions = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)
        unproj_labels = torch.full([self.max_points], -1.0, dtype=torch.int32)
        unproj_labels[:unproj_n_points] = torch.from_numpy(scan.sem_label)

        # get points and labels
        proj_range = torch.from_numpy(scan.proj_range).clone()
        proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
        proj_remission = torch.from_numpy(scan.proj_remission).clone()
        proj_mask = torch.from_numpy(scan.proj_mask)
        proj_labels = torch.from_numpy(scan.proj_sem_label).clone()
        proj_labels = proj_labels * proj_mask
        proj_x = torch.full([self.max_points], -1, dtype=torch.long)
        proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
        proj_y = torch.full([self.max_points], -1, dtype=torch.long)
        proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)
        proj = torch.cat(
            [
                proj_range.unsqueeze(0).clone(),
                proj_xyz.clone().permute(2, 0, 1),
                proj_remission.unsqueeze(0).clone(),
            ]
        )
        proj = (proj - self.sensor_img_means[:, None, None]) / self.sensor_img_stds[
            :, None, None
        ]
        proj = proj * proj_mask.float()

        # get name and sequence
        path_norm = os.path.normpath(scan_file)
        path_split = path_norm.split(os.sep)
        path_seq = path_split[-3]
        path_name = path_split[-1].replace(".bin", ".label")
        
        # return
        return (
            proj,
            proj_mask,
            proj_labels,
            unproj_labels,
            path_seq,
            path_name,
            proj_x,
            proj_y,
            proj_range,
            unproj_range,
            proj_xyz,
            unproj_xyz,
            proj_remission,
            unproj_remissions,
            unproj_n_points
        )

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
    def map(label, mapdict):
        # put label from original values to xentropy
        # or vice-versa, depending on dictionary values
        # make learning map a lookup table
        maxkey = 0
        for key, data in mapdict.items():
            if isinstance(data, list):
                nel = len(data)
            else:
                nel = 1
            if key > maxkey:
                maxkey = key
        # +100 hack making lut bigger just in case there are unknown labels
        if nel > 1:
            lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
        else:
            lut = np.zeros((maxkey + 100), dtype=np.int32)
        for key, data in mapdict.items():
            try:
                lut[key] = data
            except IndexError:
                print("Wrong key ", key)
        # do the mapping
        return lut[label]

    @staticmethod
    def collate_fn(batch):
        img, label, paths, shapes, proj = zip(*batch)
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
            torch.stack(proj, 0),
        )

