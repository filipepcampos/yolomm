import argparse
import os, sys
import math

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import pprint
import time
import torch
import torch.nn.parallel
import torch.nn.functional as F
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



def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
  # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
  x_coord = torch.arange(kernel_size)
  x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
  y_grid = x_grid.t()
  xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

  mean = (kernel_size - 1) / 2.
  variance = sigma**2.

  # Calculate the 2-dimensional gaussian kernel which is
  # the product of two gaussian distributions for two different
  # variables (in this case called x and y)
  gaussian_kernel = (1. / (2. * math.pi * variance)) *\
      torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / (2 * variance))

  # Make sure sum of values in gaussian kernel equals 1.
  gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

  # Reshape to 2d depthwise convolutional weight
  gaussian_kernel = gaussian_kernel.view(kernel_size, kernel_size)

  return gaussian_kernel


class KNN(torch.nn.Module):
  def __init__(self, params, nclasses):
    super().__init__()
    print("*"*80)
    print("Cleaning point-clouds with kNN post-processing")
    self.knn = params["knn"]
    self.search = params["search"]
    self.sigma = params["sigma"]
    self.cutoff = params["cutoff"]
    self.nclasses = nclasses
    print("kNN parameters:")
    print("knn:", self.knn)
    print("search:", self.search)
    print("sigma:", self.sigma)
    print("cutoff:", self.cutoff)
    print("nclasses:", self.nclasses)
    print("*"*80)

  def forward(self, proj_range, unproj_range, proj_argmax, px, py):
    ''' Warning! Only works for un-batched pointclouds.
        If they come batched we need to iterate over the batch dimension or do
        something REALLY smart to handle unaligned number of points in memory
    '''
    # get device
    if proj_range.is_cuda:
      device = torch.device("cuda")
    else:
      device = torch.device("cpu")

    # sizes of projection scan
    H, W = proj_range.shape

    # number of points
    P = unproj_range.shape

    # check if size of kernel is odd and complain
    if (self.search % 2 == 0):
      raise ValueError("Nearest neighbor kernel must be odd number")

    # calculate padding
    pad = int((self.search - 1) / 2)

    # unfold neighborhood to get nearest neighbors for each pixel (range image)
    proj_unfold_k_rang = F.unfold(proj_range[None, None, ...],
                                  kernel_size=(self.search, self.search),
                                  padding=(pad, pad))

    # index with px, py to get ALL the pcld points
    idx_list = py * W + px
    unproj_unfold_k_rang = proj_unfold_k_rang[:, :, idx_list]

    # WARNING, THIS IS A HACK
    # Make non valid (<0) range points extremely big so that there is no screwing
    # up the nn self.search
    unproj_unfold_k_rang[unproj_unfold_k_rang < 0] = float("inf")

    # now the matrix is unfolded TOTALLY, replace the middle points with the actual range points
    center = int(((self.search * self.search) - 1) / 2)
    unproj_unfold_k_rang[:, center, :] = unproj_range

    # now compare range
    k2_distances = torch.abs(unproj_unfold_k_rang - unproj_range)

    # make a kernel to weigh the ranges according to distance in (x,y)
    # I make this 1 - kernel because I want distances that are close in (x,y)
    # to matter more
    inv_gauss_k = (
        1 - get_gaussian_kernel(self.search, self.sigma, 1)).view(1, -1, 1)
    inv_gauss_k = inv_gauss_k.to(device).type(proj_range.type())

    # apply weighing
    k2_distances = k2_distances * inv_gauss_k

    # find nearest neighbors
    _, knn_idx = k2_distances.topk(
        self.knn, dim=1, largest=False, sorted=False)

    # do the same unfolding with the argmax
    proj_unfold_1_argmax = F.unfold(proj_argmax[None, None, ...].float(),
                                    kernel_size=(self.search, self.search),
                                    padding=(pad, pad)).long()
    unproj_unfold_1_argmax = proj_unfold_1_argmax[:, :, idx_list]

    # get the top k predictions from the knn at each pixel
    knn_argmax = torch.gather(
        input=unproj_unfold_1_argmax, dim=1, index=knn_idx)

    # fake an invalid argmax of classes + 1 for all cutoff items
    if self.cutoff > 0:
      knn_distances = torch.gather(input=k2_distances, dim=1, index=knn_idx)
      knn_invalid_idx = knn_distances > self.cutoff
      knn_argmax[knn_invalid_idx] = self.nclasses

    # now vote
    # argmax onehot has an extra class for objects after cutoff
    knn_argmax_onehot = torch.zeros(
        (1, self.nclasses + 1, P[0]), device=device).type(proj_range.type())
    ones = torch.ones_like(knn_argmax).type(proj_range.type())
    knn_argmax_onehot = knn_argmax_onehot.scatter_add_(1, knn_argmax, ones)

    # now vote (as a sum over the onehot shit)  (don't let it choose unlabeled OR invalid)
    knn_argmax_out = knn_argmax_onehot[:, 1:-1].argmax(dim=1) + 1

    # reshape again
    knn_argmax_out = knn_argmax_out.view(P)

    return knn_argmax_out






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

    checkpoint_file = "runs/MultimodalKITTIDatasetLIDAR/_2023-05-13-21-14/epoch-159.pth"
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

    post_params = {
        "knn": 5,
        "search": 5,
        "sigma": 1.0,
        "cutoff": 1.0,
    }

    post = KNN(post_params, 20)

    # train for one epoch
    test_semantic_kitti(
        cfg,
        dataset,
        valid_loader,
        model,
        logger,
        device,
        data["learning_ignore"],
        post,
        data["labels"],
    )


if __name__ == "__main__":
    main()
