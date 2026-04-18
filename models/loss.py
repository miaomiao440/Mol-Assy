# -*- coding: utf-8 -*-
# @Author  : LG
from torch import nn
import torch
from torch.nn import functional as F



def binary_cross_entropy(predict, target, weight=None, size_average=True):
    # predict: (batch,)
    # target:  (batch,)
    # weight:  (batch,)

    eps = 1e-10
    loss = target * torch.log(predict + eps) + (1 - target) * torch.log(1 - predict + eps)
    loss = -loss

    if weight is not None:
        loss = loss * weight

    return loss.mean() if size_average else loss.sum()


