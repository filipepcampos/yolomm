import argparse
import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from lib.config import cfg
from lib.dataset.road_kitti import KITTIRoadDataset
from lib.dataset.bdd import BddDataset
import torchvision.transforms as transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

dataset = KITTIRoadDataset(
    cfg,
    True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    ),
)

import matplotlib.pyplot as plt
import cv2


img, target, path, _ = dataset[0]


plt.imshow(img.transpose(0, 2).transpose(0, 1))
plt.show()
plt.imshow(target[0])
plt.show()
plt.imshow(target[1])
plt.show()


# dataset = BddDataset(cfg,
#     True,
#     inputsize=cfg.MODEL.IMAGE_SIZE,
#     transform=transforms.Compose(
#         [
#             transforms.ToTensor(),
#             normalize,
#         ]
#     ),)

# img, target, path, _ = dataset[0]
# masks = target[1]
# plt.imshow(img.transpose(0,2).transpose(0,1))
# plt.show()
# plt.imshow(masks[0])
# plt.show()
# plt.imshow(masks[1])
# plt.show()

# Primeira: drivable area a roxo, background a amarelo
# Segunda: drivable amarelo, background roxo
