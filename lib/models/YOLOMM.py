import torch
from torch import tensor
import torch.nn as nn
import sys, os
import math
import sys

sys.path.append(os.getcwd())
# sys.path.append("lib/models")
# sys.path.append("lib/utils")
# sys.path.append("/workspace/wh/projects/DaChuang")
from lib.utils import initialize_weights

# from lib.models.common2 import DepthSeperabelConv2d as Conv
# from lib.models.common2 import SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect
from lib.models.common import (
    Conv,
    WidthConv,
    SPP,
    Bottleneck,
    BottleneckCSP,
    Focus,
    Concat,
    Detect,
    SharpenConv,
    Add,
)
from torch.nn import Upsample
from lib.utils import check_anchor_order
from lib.core.evaluate import SegmentationMetric
from lib.utils.utils import time_synchronized

# The lane line and the driving area segment branches without share information with each other and without link
YOLOMM = [
    [33, 41, 51],  # Det_out_idx, Da_Segout_idx, LL_Segout_idx
    [0, 8], # Proj_idx, Img_idx
    [-1, WidthConv, [5, 16, 3, 2]],  # 0 start of proj branch
    [-1, BottleneckCSP, [16, 16, 1]],  # 1
    [-1, WidthConv, [16, 32, 3, 2]],  # 2
    [-1, BottleneckCSP, [32, 32, 1]],  # 3
    [-1, WidthConv, [32, 64, 3, 2]],  # 4 
    [-1, BottleneckCSP, [64, 64, 1]], # 5
    [-1, WidthConv, [64, 64, 3, 2]],  # 6
    [-1, BottleneckCSP, [64, 64, 1]],  # 7 end of proj branch
    [-1, Focus, [3, 32, 3]],  # 8 start of img branch
    [-1, Conv, [32, 64, 3, 2]],  # 9
    [-1, BottleneckCSP, [64, 64, 1]],  # 10
    [[-1, 7], Add, []], # 11 join both branches
    [-1, Conv, [64, 128, 3, 2]],  # 12
    [-1, BottleneckCSP, [128, 128, 3]],  # 13
    [-1, Conv, [128, 256, 3, 2]],  # 14
    [-1, BottleneckCSP, [256, 256, 3]],  # 15
    [-1, Conv, [256, 512, 3, 2]],  # 16
    [-1, SPP, [512, 512, [5, 9, 13]]],  # 17
    [-1, BottleneckCSP, [512, 512, 1, False]],  # 18
    [-1, Conv, [512, 256, 1, 1]],  # 19
    [-1, Upsample, [None, 2, "nearest"]],  # 20
    [[-1, 15], Concat, [1]],  # 21
    [-1, BottleneckCSP, [512, 256, 1, False]],  # 22
    [-1, Conv, [256, 128, 1, 1]],  # 23
    [-1, Upsample, [None, 2, "nearest"]],  # 24
    [[-1, 13], Concat, [1]],  # 25         #Encoder
    [-1, BottleneckCSP, [256, 128, 1, False]],  # 26
    [-1, Conv, [128, 128, 3, 2]],  # 27
    [[-1, 23], Concat, [1]],  # 28
    [-1, BottleneckCSP, [256, 256, 1, False]],  # 29
    [-1, Conv, [256, 256, 3, 2]],  # 30
    [[-1, 19], Concat, [1]],  # 31
    [-1, BottleneckCSP, [512, 512, 1, False]],  # 32
    [
        [26, 29, 32],
        Detect,
        [
            1,
            [[3, 9, 5, 11, 4, 20], [7, 18, 6, 39, 12, 31], [19, 50, 38, 81, 68, 157]],
            [128, 256, 512],
        ],
    ],  # Detection head 33
    [25, Conv, [256, 128, 3, 1]],  # 34
    [-1, Upsample, [None, 2, "nearest"]],  # 35
    [-1, BottleneckCSP, [128, 64, 1, False]],  # 36
    [-1, Conv, [64, 32, 3, 1]],  # 37
    [-1, Upsample, [None, 2, "nearest"]],  # 38
    [-1, Conv, [32, 16, 3, 1]],  # 39
    [-1, BottleneckCSP, [16, 8, 1, False]],  # 40
    [-1, Upsample, [None, 2, "nearest"]],  # 41
    [-1, Conv, [8, 2, 3, 1]],  # 42 Driving area segmentation head
    [25, Conv, [256, 128, 3, 1]],  # 43
    [-1, Upsample, [None, 2, "nearest"]],  # 44
    [-1, BottleneckCSP, [128, 64, 1, False]],  # 45
    [-1, Conv, [64, 32, 3, 1]],  # 46
    [-1, Upsample, [None, (1, 2), "nearest"]],  # 47
    [-1, Conv, [32, 16, 3, 1]],  # 48
    [-1, BottleneckCSP, [16, 8, 1, False]],  # 49
    [-1, Upsample, [None, (1, 2), "nearest"]],  # 50
    [-1, Conv, [8, 8, 3, 1]],  # 51
    [-1, Upsample, [None, (1, 2), "nearest"]],  # 52
    [-1, Conv, [8, 8, 3, 1]],  # 53
    [-1, Upsample, [None, (1, 2), "nearest"]],  # 54
    [-1, Conv, [8, 2, 3, 1]],  # 55 Lidar segmentation head
]


class MCnet(nn.Module):
    def __init__(self, block_cfg, **kwargs):
        super(MCnet, self).__init__()
        layers, save = [], []
        self.nc = 1
        self.detector_index = -1
        self.det_out_idx = block_cfg[0][0]
        self.seg_out_idx = block_cfg[0][1:]
        self.proj_idx = block_cfg[1][0]
        self.img_idx = block_cfg[1][1]

        # Build model
        for i, (from_, block, args) in enumerate(block_cfg[2:]):
            block = eval(block) if isinstance(block, str) else block  # eval strings
            if block is Detect:
                self.detector_index = i
            block_ = block(*args)
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            save.extend(
                x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1
            )  # append to savelist
        # assert self.detector_index == block_cfg[0][0]

        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self.names = [str(i) for i in range(self.nc)]

        # set stride、anchor for detector
        Detector = self.model[self.detector_index]  # detector
        if isinstance(Detector, Detect):
            s = 128  # 2x min stride
            # for x in self.forward(torch.zeros(1, 3, s, s)):
            #     print (x.shape)
            with torch.no_grad():
                model_out = self.forward(torch.zeros(1, 3, s, s), torch.zeros(1, 5, int(s/4), s*4))
                detects, _, _ = model_out
                Detector.stride = torch.tensor(
                    [s / x.shape[-2] for x in detects]
                )  # forward
            # print("stride"+str(Detector.stride ))
            Detector.anchors /= Detector.stride.view(
                -1, 1, 1
            )  # Set the anchors for the corresponding scale
            check_anchor_order(Detector)
            self.stride = Detector.stride
            self._initialize_biases()

        initialize_weights(self)

    def forward(self, img, proj):
        cache = []
        out = []
        det_out = None
        Da_fmap = []
        LL_fmap = []
        for i, block in enumerate(self.model):
            if block.from_ != -1:
                img = (
                    cache[block.from_]
                    if isinstance(block.from_, int)
                    else [img if j == -1 else cache[j] for j in block.from_]
                )  # calculate concat detect

            if i < self.img_idx:
                proj = block(proj)
                cache.append(proj if block.index in self.save else None)
            else:
                img = block(img)
                cache.append(img if block.index in self.save else None)
            
                if i in self.seg_out_idx:  # save driving area segment result
                    m = nn.Sigmoid()
                    out.append(m(img))
                if i == self.detector_index:
                    det_out = img
        out.insert(0, det_out)
        return out

    def _initialize_biases(
        self, cf=None
    ):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.model[self.detector_index]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(
                8 / (640 / s) ** 2
            )  # obj (8 objects per 640 image)
            b.data[:, 5:] += (
                math.log(0.6 / (m.nc - 0.99))
                if cf is None
                else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


def get_net(cfg, **kwargs):
    m_block_cfg = YOLOMM
    model = MCnet(m_block_cfg, **kwargs)
    return model


if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter

    model = get_net(False)
    input_ = torch.randn((1, 3, 256, 256))
    gt_ = torch.rand((1, 2, 256, 256))
    metric = SegmentationMetric(2)
    model_out, SAD_out = model(input_)
    detects, dring_area_seg, lane_line_seg = model_out
    Da_fmap, LL_fmap = SAD_out
    for det in detects:
        print(det.shape)
    print(dring_area_seg.shape)
    print(lane_line_seg.shape)
