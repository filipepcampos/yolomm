import argparse
import os, sys
import math

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import pprint
import time
import torch
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from lib.models import get_net
from lib.config import cfg
from lib.config import update_config
from lib.models import get_net


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multitask network")
    # general
    # parser.add_argument('--cfg',
    #                     help='experiment configure file name',
    #                     required=True,
    #                     type=str)

    # philly
    parser.add_argument("--modelDir", help="model directory", type=str, default="")
    parser.add_argument("--logDir", help="log directory", type=str, default="runs/")
    parser.add_argument("--dataDir", help="data directory", type=str, default="")
    parser.add_argument(
        "--prevModelDir", help="prev Model directory", type=str, default=""
    )

    parser.add_argument(
        "--sync-bn",
        action="store_true",
        help="use SyncBatchNorm, only available in DDP mode",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="DDP parameter, do not modify"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.001, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.6, help="IOU threshold for NMS"
    )
    args = parser.parse_args()

    return args


def get_cfg():
    # set all the configurations
    args = parse_args()
    update_config(cfg, args)
    return cfg

cfg = get_cfg()
model = get_net(cfg)
input_img = torch.randn((1, 3, 256, 512))
input_proj = torch.randn((1, 5, 64, 2048))
print(input_img.shape)
print("________________\n\n\n\n")
output = model(input_img, input_proj)

