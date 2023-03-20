import argparse
import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from lib.config import cfg
from lib.dataset.kitti import KITTIDataset
import torchvision.transforms as transforms

normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

dataset = KITTIDataset(cfg, True, transform=transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        ))
