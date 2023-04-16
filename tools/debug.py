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
from torch.cuda import amp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import numpy as np
import yaml
from lib.utils import DataLoaderX, torch_distributed_zero_first
from tensorboardX import SummaryWriter

from lib.models import get_net
import lib.dataset as dataset
from lib.dataset.semantic_kitti import Parser
from lib.config import cfg
from lib.config import update_config
from lib.core.loss import get_kitti_da_loss
from lib.core.function import train_semantic_kitti
from lib.core.function import validate_semantic_kitti
from lib.core.general import fitness
from lib.models import get_net
from lib.utils import is_parallel
from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger, select_device
from lib.utils import run_anchor

import torchsummary

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
torchsummary.summary(model, (3, 224, 640))

