import argparse
import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from lib.config import cfg
from lib.dataset.kitti import KITTIDataset

KITTIDataset(cfg, True)