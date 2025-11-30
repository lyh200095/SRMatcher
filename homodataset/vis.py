

from matplotlib import pyplot as plt
import torch
from torch import Tensor
import numpy as np

def convert_format(x):
    if isinstance(x, Tensor):
        x = x.cpu().numpy()

    # convert x to B, H, W, C
    sz = x.shape
    if len(sz) == 2:
        x = x.reshape(1, sz[0], sz[1], 1)
    elif len(sz) == 3:
        if sz[-1] == 3:  # RGB
            x = x.reshape(1, sz[0], sz[1], sz[2])
        else:
            x = x.reshape(sz[0], sz[1], sz[2])

    if x.shape[3] == 1:
        x = x.repeat(x, 3, 3)
    return x


def imshow(x: np.array, numcol, seph=None, sepw=None):
    """

    Args:
        x: 输入numpy 格式的tensor: B, H, W, C  # C为RGB
        numcol: 每列显示的图片数量
        seph: height 之间的分割
        sepw: width 之间的分割

    Returns:

    """
    x = convert_format(x)

