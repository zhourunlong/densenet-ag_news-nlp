import fire
import os
import time
import torch
from dataset import *
from torchvision import datasets, transforms
from models import DenseNet
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def demo(depth=58, growth_rate=12, efficient=False):
    # Get densenet configuration
    if (depth - 4) % 3:
        raise Exception('Invalid depth')
    block_config = [(depth - 4) // 6 for _ in range(3)]
    model = DenseNet(
        growth_rate=growth_rate,
        block_config=block_config,
        num_classes=4,
        small_inputs=True,
        efficient=efficient,
    )
    model.load_state_dict(torch.load(os.path.join('./ckpt2/model.dat')))
    t = list(model.state_dict())
    n = len(t)
    w = []
    for x in range(3):
        w.append([])
        for i in range(9):
            w[x].append([])
            for j in range(9):
                w[x][i].append(0)
    for name in model.state_dict():
        if len(name) == 49 and name[37] == 'c':
            x, i, j = int(name[19]), int(name[32]), int(name[34])
            a = abs(model.state_dict()[name])
            w[x - 1][j][i - 1] = a.sum()
    for x in range(3):
        for i in range(9):
            mx = 0
            for j in range(i, 9):
                mx = max(mx, w[x][i][j])
            for j in range(i, 9):
                w[x][i][j] = w[x][i][j] / mx
    mask = []
    for i in range(9):
        mask.append([])
        for j in range(9):
            mask[i].append(j > i)
    ax = []
    for x in range(3):
        sns.set()
        ax.append(sns.heatmap(w[x], vmin = 0, vmax = 1, cmap = 'jet', square = True, mask = mask))
        ax[x].set_title('Dense Block %s' % (x + 1))
        ax[x].set_xlabel('Target layer (l)', fontsize=15)
        ax[x].set_ylabel('Source layer (s)', fontsize=15)
        plt.show(ax[x])

if __name__ == '__main__':
    demo()
