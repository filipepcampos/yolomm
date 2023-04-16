import argparse
import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from lib.config import cfg
from lib.dataset.road_kitti import KITTIRoadDataset
from lib.dataset.object_kitti import KITTIDataset
from lib.dataset.bdd import BddDataset
import torchvision.transforms as transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# dataset = KITTIRoadDataset(
#     cfg,
#     True,
#     transform=transforms.Compose(
#         [
#             transforms.ToTensor(),
#             normalize,
#         ]
#     ),
# )

dataset = KITTIDataset(
    cfg,
    True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    ),
)

print(dataset[0][1])
