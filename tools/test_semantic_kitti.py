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
from lib.utils import DataLoaderX
from tensorboardX import SummaryWriter

import lib.dataset as dataset
from lib.config import cfg
from lib.config import update_config
from lib.core.loss import get_kitti_lidar_loss
from lib.core.function import train_semantic_kitti
from lib.core.function import test_semantic_kitti
from lib.dataset.MultimodalKITTIDatasetLIDAR import MultimodalKITTIDatasetLIDAR
from lib.models import get_net
from lib.utils import is_parallel
from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger, select_device
from lib.utils import run_anchor


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


def override_config(cfg):
    """
    A quick trick to ensure the correct config is always applied
    """
    cfg.defrost()
    cfg.DATASET.DATACFG = "/home/up201905609/yolomm/lib/config/semantic_kitti_v2.yaml"
    cfg.DATASET.DATASET = "MultimodalKITTIDatasetLIDAR"
    cfg.DATASET.DATA_FORMAT = "png"
    cfg.DATASET.SELECT_DATA = False
    cfg.DATASET.ORG_IMG_SIZE = [512, 1382]
    cfg.MODEL.IMAGE_SIZE = [512, 256]  # width * height, ex: 192 * 256
    cfg.num_seg_class = 13
    cfg.freeze()


def main():
    # set all the configurations
    args = parse_args()
    update_config(cfg, args)
    override_config(cfg)

    # Set DDP variables
    world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    global_rank = int(os.environ["RANK"]) if "RANK" in os.environ else -1

    rank = global_rank
    # print(rank)
    # TODO: handle distributed training logger
    # set the logger, tb_log_dir means tensorboard logdir

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, cfg.LOG_DIR, "train", rank=rank
    )

    if rank in [-1, 0]:
        logger.info(pprint.pformat(args))
        logger.info(cfg)

        writer_dict = {
            "writer": SummaryWriter(log_dir=tb_log_dir),
            "train_global_steps": 0,
            "valid_global_steps": 0,
        }
    else:
        writer_dict = None

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # bulid up model
    # start_time = time.time()
    print("begin to build up model...")
    # DP mode
    device = (
        select_device(logger, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS))
        if not cfg.DEBUG
        else select_device(logger, "cpu")
    )

    if args.local_rank != -1:
        assert torch.cuda.device_count() > args.local_rank
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(
            backend="nccl", init_method="env://"
        )  # distributed backend

    print("load model to device")
    model = get_net(cfg).to(device)

    checkpoint_file = "runs/MultimodalKITTIDatasetLIDAR/_2023-05-09-02-23/epoch-46.pth"
    model_dict = model.state_dict()
    logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
    checkpoint = torch.load(checkpoint_file)
    checkpoint_dict = checkpoint["state_dict"]
    # checkpoint_dict = {k: v for k, v in checkpoint['state_dict'].items() if k.split(".")[1] in det_idx_range}
    model_dict.update(checkpoint_dict)
    model.load_state_dict(model_dict)
    logger.info("=> loaded checkpoint '{}' ".format(checkpoint_file))

    # assign model params
    model.gr = 1.0
    model.nc = 1
    # print('bulid model finished')

    print("begin to load data")
    # Data loading
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    print("dataset, ", cfg.DATASET.DATASET)
    data = yaml.safe_load(open(cfg.DATASET.DATACFG, 'r'))

    dataset = eval("dataset." + cfg.DATASET.DATASET)(
        cfg=cfg,
        is_train=True,
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        ),
        labels=data["labels"],
        color_map=data["color_map"],
        learning_map=data["learning_map"],
        learning_map_inv=data["learning_map_inv"],
        sensor=data["sensor"],
        max_points=150000,
        sequences=data["split"]["train"]
    )

    TRAIN_SIZE = 0.8
    train_dataset = torch.utils.data.Subset(
        dataset, range(0, int(len(dataset)*TRAIN_SIZE), 1)
    )

    if rank in [-1, 0]:
        valid_loader = torch.utils.data.Subset(
            dataset, range(int(TRAIN_SIZE*len(dataset)), len(dataset), 1)
        )

        valid_loader = DataLoaderX(
            valid_loader,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            # collate_fn=dataset.collate_fn,
        )
        print("load data finished")

    if rank in [-1, 0]:
        if cfg.NEED_AUTOANCHOR:
            logger.info("begin check anchors")
            run_anchor(model,
                logger,
                train_dataset,
                model=model,
                thr=cfg.TRAIN.ANCHOR_THRESHOLD,
                imgsz=min(cfg.MODEL.IMAGE_SIZE),
            )
        else:
            logger.info("anchors loaded successfully")
            det = (
                model.module.model[model.module.detector_index]
                if is_parallel(model)
                else model.model[model.detector_index]
            )
            logger.info(str(det.anchors))

    # training
    scaler = amp.GradScaler(enabled=device.type != "cpu")
    print("=> start training...")

    # train for one epoch
    test_semantic_kitti(
        cfg,
        dataset,
        valid_loader,
        model,
        logger,
        device,
        data["learning_ignore"]
    )


if __name__ == "__main__":
    main()
