#!/usr/bin/env python3

# This file is covered by the LICENSE file in the root of this project.

import torch
import numpy as np
import time
import torch.nn as nn
import torch.nn.functional as F
import __init__ as booger

class oneHot(nn.Module):
  def __init__(self, device, nclasses, spatial_dim=2):
    super().__init__()
    self.device = device
    self.nclasses = nclasses
    self.spatial_dim = spatial_dim

  def onehot1dspatial(self, x):
    # we only do tensors that 1d tensors that are batched or not, so check
    assert(len(x.shape) == 1 or len(x.shape) == 2)
    # if not batched, batch
    remove_dim = False  # flag to unbatch
    if len(x.shape) == 1:
      # add batch dimension
      x = x[None, ...]
      remove_dim = True

    # get tensor shape
    n, b = x.shape

    # scatter to onehot
    one_hot = torch.zeros((n, self.nclasses, b),
                          device=self.device).scatter_(1, x.unsqueeze(1), 1)

    # x is now [n,classes,b]

    # if it used to be unbatched, then unbatch it
    if remove_dim:
      one_hot = one_hot[0]

    return one_hot

  def onehot2dspatial(self, x):
    # we only do tensors that 2d tensors that are batched or not, so check
    assert(len(x.shape) == 2 or len(x.shape) == 3)
    # if not batched, batch
    remove_dim = False  # flag to unbatch
    if len(x.shape) == 2:
      # add batch dimension
      x = x[None, ...]
      remove_dim = True

    # get tensor shape
    n, h, w = x.shape

    # scatter to onehot
    one_hot = torch.zeros((n, self.nclasses, h, w),
                          device=self.device).scatter_(1, x.unsqueeze(1), 1)

    # x is now [n,classes,b]

    # if it used to be unbatched, then unbatch it
    if remove_dim:
      one_hot = one_hot[0]

    return one_hot

  def forward(self, x):
    # do onehot here
    if self.spatial_dim == 1:
      return self.onehot1dspatial(x)
    elif self.spatial_dim == 2:
      return self.onehot2dspatial(x)


class borderMask(nn.Module):
  def __init__(self, nclasses, device, border_size, kern_conn=4, background_class=None):
    """Get the binary border mask of a labeled 2d range image.

  Args:
      nclasses(int)         : The number of classes labeled in the input image
      device(torch.device)  : Process in host or cuda?
      border_size(int)      : How many erode iterations to perform for the mask
      kern_conn(int)        : The connectivity kernel number (4 or 8)
      background_class(int) : "unlabeled" class in dataset (to avoid double borders)

  Returns:
      eroded_output(tensor) : The 2d binary border mask, 1 where a intersection
                              between classes occurs, 0 everywhere else

  """
    super().__init__()
    self.nclasses = nclasses
    self.device = device
    self.border_size = border_size
    self.kern_conn = kern_conn
    self.background_class = background_class
    if self.background_class is not None:
      self.include_idx = list(range(self.nclasses))
      self.exclude_idx = self.include_idx.pop(self.background_class)

    # check connectivity
    # For obtaining the border mask we will be eroding the input image, for this
    # reason we only support erode_kernels with connectivity 4 or 8
    assert self.kern_conn in (4, 8), ("The specified kernel connectivity(kern_conn= %r) is "
                                      "not supported" % self.kern_conn)

    # make the onehot inferer
    self.onehot = oneHot(self.device,
                         self.nclasses,
                         spatial_dim=2)  # range labels

  def forward(self, range_label):
    # length of shape of range_label must be 3 (N, H, W)
    must_unbatch = False  # remove batch dimension after operation?
    if len(range_label.shape) != 3:
      range_label = range_label[None, ...]
      must_unbatch = True

    # The range_label comes labeled, we need to create one tensor per class, thus:
    input_tensor = self.onehot(range_label)  # (N, C, H, W)

    # Because we are using GT range_labels, there is a lot of pixels that end up
    # unlabeled(thus, in the background). If we feed the erosion algorithm with
    # this "raw" gt_labels we will detect intersection between the other classes
    # and the backgorund, and we will end with the incorrect border mask. To solve
    # this issue we need to pre process the input gt_label. The artifact in this
    # case will be to sum the background channel(mostly the channel 0) to
    # all the rest channels expect for the background channel itself.
    # This will allow us to avoid detecting intersections between a class and the
    # background. This also force us to change the logical AND we were doing to
    # obtain the border mask when we were working with predicted labels.
    # With predicted labels won't see this problem because all the pixels belongs
    # to at least one class
    if self.background_class is not None:
      input_tensor[:, self.include_idx] = input_tensor[:, self.include_idx] + \
          input_tensor[:, self.exclude_idx]

    # C denotes a number of channels, N, H and W are dismissed
    C = input_tensor.shape[1]

    # Create an empty erode kernel and send it to 'device'
    erode_kernel = torch.zeros((C, 1, 3, 3), device=self.device)
    if self.kern_conn == 4:
      erode_kernel[:] = torch.tensor([[0, 1, 0],
                                      [1, 1, 1],
                                      [0, 1, 0]], device=self.device)
    else:
      erode_kernel[:] = torch.tensor([[1, 1, 1],
                                      [1, 1, 1],
                                      [1, 1, 1]], device=self.device)

    # to check connectivity
    kernel_sum = erode_kernel[0][0].sum()  # should be kern_conn + 1

    # erode the input image border_size times
    erode_input = input_tensor
    for _ in range(self.border_size):
      eroded_output = F.conv2d(erode_input, erode_kernel, groups=C, padding=1)
      # Pick the elements that match the kernel_sum to obtain the eroded
      # output and convert to dtype=float32
      eroded_output = (eroded_output == kernel_sum).float()
      erode_input = eroded_output

    # We want to sum up all the channels into 1 unique border mask
    # Even when we added the background to all the rest of the channels, there
    # might be "bodies" in the background channel, thus, the erosion process can
    # output "false positives" were this "bodies" are present in the background.
    # We need to obtain the background mask and add it to the eroded bodies to
    # obtain a consisent output once we calculate the border mask
    if self.background_class is not None:
      background_mask = (eroded_output[:, self.exclude_idx] == 1)

    # The eroded_bodies mask will consist in all the pixels were the convolution
    # returned 1 for all the channels, therefore we need to sum up all the
    # channels into one unique tensor and add the background mask to avoid having
    # the background in the border mask output
    eroded_bodies = (eroded_output.sum(1, keepdim=True) == 1)
    if self.background_class is not None:
      eroded_bodies = eroded_bodies + background_mask

    # we want the opposite
    borders = 1 - eroded_bodies

    # unbatch?
    if must_unbatch:
      borders = borders[0]
      # import cv2
      # import numpy as np
      # bordersprint = (borders * 255).squeeze().cpu().numpy().astype(np.uint8)
      # cv2.imshow("border", bordersprint)
      # cv2.waitKey(0)

    return borders

class iouEval:
  def __init__(self, n_classes, device, ignore=None):
    self.n_classes = n_classes
    self.device = device
    # if ignore is larger than n_classes, consider no ignoreIndex
    self.ignore = torch.tensor(ignore).long()
    self.include = torch.tensor(
        [n for n in range(self.n_classes) if n not in self.ignore]).long()
    print("[IOU EVAL] IGNORE: ", self.ignore)
    print("[IOU EVAL] INCLUDE: ", self.include)
    self.reset()

  def num_classes(self):
    return self.n_classes

  def reset(self):
    self.conf_matrix = torch.zeros(
        (self.n_classes, self.n_classes), device=self.device).long()
    self.ones = None
    self.last_scan_size = None  # for when variable scan size is used

  def addBatch(self, x, y):  # x=preds, y=targets
    # if numpy, pass to pytorch
    # to tensor
    if isinstance(x, np.ndarray):
      x = torch.from_numpy(np.array(x)).long().to(self.device)
    if isinstance(y, np.ndarray):
      y = torch.from_numpy(np.array(y)).long().to(self.device)

    # sizes should be "batch_size x H x W"
    x_row = x.reshape(-1)  # de-batchify
    y_row = y.reshape(-1)  # de-batchify

    # idxs are labels and predictions
    idxs = torch.stack([x_row, y_row], dim=0)

    # ones is what I want to add to conf when I
    if self.ones is None or self.last_scan_size != idxs.shape[-1]:
      self.ones = torch.ones((idxs.shape[-1]), device=self.device).long()
      self.last_scan_size = idxs.shape[-1]

    # make confusion matrix (cols = gt, rows = pred)
    self.conf_matrix = self.conf_matrix.index_put_(
        tuple(idxs), self.ones, accumulate=True)

    # print(self.tp.shape)
    # print(self.fp.shape)
    # print(self.fn.shape)

  def getStats(self):
    # remove fp and fn from confusion on the ignore classes cols and rows
    conf = self.conf_matrix.clone().double()
    conf[self.ignore] = 0
    conf[:, self.ignore] = 0

    # get the clean stats
    tp = conf.diag()
    fp = conf.sum(dim=1) - tp
    fn = conf.sum(dim=0) - tp
    return tp, fp, fn

  def getIoU(self):
    tp, fp, fn = self.getStats()
    intersection = tp
    union = tp + fp + fn + 1e-15
    iou = intersection / union
    iou_mean = (intersection[self.include] / union[self.include]).mean()
    return iou_mean, iou  # returns "iou mean", "iou per class" ALL CLASSES

  def getacc(self):
    tp, fp, fn = self.getStats()
    total_tp = tp.sum()
    total = tp[self.include].sum() + fp[self.include].sum() + 1e-15
    acc_mean = total_tp / total
    return acc_mean  # returns "acc mean"


class biouEval(iouEval):
  def __init__(self, n_classes, device, ignore=None, border_size=1, kern_conn=4):
    super().__init__(n_classes, device, ignore)
    self.border_size = border_size
    self.kern_conn = kern_conn

    # check that I am only ignoring one class
    if len(ignore) > 1:
      raise ValueError("Length of ignored class list should be 1 or 0")
    elif len(ignore) == 0:
      ignore = None
    else:
      ignore = ignore[0]

    self.borderer = borderMask(self.n_classes, self.device,
                               self.border_size, self.kern_conn,
                               background_class=ignore)
    self.reset()

  def reset(self):
    super().reset()
    return

  def addBorderBatch1d(self, range_y, x, y, px, py):
    '''range_y=target as img, x=preds, y=targets, px,py=idxs of points of
       pointcloud in range img
       WARNING: Only batch size 1 works for now
    '''
    # if numpy, pass to pytorch
    # to tensor
    if isinstance(range_y, np.ndarray):
      range_y = torch.from_numpy(np.array(range_y)).long().to(self.device)
    if isinstance(x, np.ndarray):
      x = torch.from_numpy(np.array(x)).long().to(self.device)
    if isinstance(y, np.ndarray):
      y = torch.from_numpy(np.array(y)).long().to(self.device)
    if isinstance(px, np.ndarray):
      px = torch.from_numpy(np.array(px)).long().to(self.device)
    if isinstance(py, np.ndarray):
      py = torch.from_numpy(np.array(py)).long().to(self.device)

    # get border mask of range_y
    border_mask_2d = self.borderer(range_y)

    # filter px, py according to if they are on border mask or not
    border_mask_1d = border_mask_2d[0, py, px].byte()

    # get proper points from filtered x and y
    x_in_mask = torch.masked_select(x, border_mask_1d)
    y_in_mask = torch.masked_select(y, border_mask_1d)

    # add batch
    self.addBatch(x_in_mask, y_in_mask)


if __name__ == "__main__":
  # mock problem
  nclasses = 2
  ignore = []

  # test with 2 squares and a known IOU
  lbl = torch.zeros((7, 7)).long()
  argmax = torch.zeros((7, 7)).long()

  # put squares
  lbl[2:4, 2:4] = 1
  argmax[3:5, 3:5] = 1

  # make evaluator
  eval = iouEval(nclasses, torch.device('cpu'), ignore)

  # run
  eval.addBatch(argmax, lbl)
  m_iou, iou = eval.getIoU()
  print("*"*80)
  print("Small iou mock problem")
  print("IoU: ", m_iou)
  print("IoU class: ", iou)
  m_acc = eval.getacc()
  print("Acc: ", m_acc)
  print("*"*80)
