import os

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff


def A(image_id):
    filename = os.path.join('..', 'data', 'sixteen_band', '{}_A.tif'.format(image_id))
    img = tiff.imread(filename)
    # print(img.shape)
    # 将channel数放置到最后一位上
    img = np.rollaxis(img, 0, 3)
    return img


def M(image_id):
    filename = os.path.join('..', 'data', 'sixteen_band', '{}_M.tif'.format(image_id))
    img = tiff.imread(filename)
    # print(img.shape)
    img = np.rollaxis(img, 0, 3)
    return img


def P(image_id):
    filename = os.path.join('..', 'data', 'sixteen_band', '{}_P.tif'.format(image_id))
    img = tiff.imread(filename)
    # P段是灰度图，只有一个channel，所以不需要交换轴
    return img


def RGB(image_id):
    filename = os.path.join('..', 'data', 'three_band', '{}.tif'.format(image_id))
    img = tiff.imread(filename)
    # print(img.shape)
    img = np.rollaxis(img, 0, 3)
    return img


# 因为遥感图像数据位数是16位(uint16),需要使用5%的线性拉伸,不然显示起来不正常
def stretch_n(bands, lower_percent=5, higher_percent=95):
    # print(bands.dtype)
    # 一定要使用float32类型，原因有两个：1、Keras不支持float64运算；2、float32运算要好于uint16
    out = np.zeros_like(bands).astype(np.float32)
    # print(out.dtype)
    for i in range(bands.shape[2]):
        a = 0  # np.min(band)
        b = 1  # np.max(band)
        # 计算百分位数（从小到大排序之后第 percent% 的数）
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t
    # print(out.dtype)
    return out


def display_img(img):
    # P
    if img.ndim == 2:
        plt.imshow(img)
        plt.show()
    # RGB
    elif img.shape[2] == 3:
        plt.imshow(img)
        plt.show()
    # M or A
    elif img.shape[2] == 8:
        # 将画布分成4行2列，刚好每个格子放一个channel
        fig = plt.figure()
        ax = fig.add_subplot(4, 2, 1)
        ax.imshow(img[:, :, 0])
        ax1 = fig.add_subplot(4, 2, 2)
        ax1.imshow(img[:, :, 1])
        ax2 = fig.add_subplot(4, 2, 3)
        ax2.imshow(img[:, :, 2])
        ax3 = fig.add_subplot(4, 2, 4)
        ax3.imshow(img[:, :, 3])
        ax4 = fig.add_subplot(4, 2, 5)
        ax4.imshow(img[:, :, 4])
        ax5 = fig.add_subplot(4, 2, 6)
        ax5.imshow(img[:, :, 5])
        ax6 = fig.add_subplot(4, 2, 7)
        ax6.imshow(img[:, :, 6])
        ax7 = fig.add_subplot(4, 2, 8)
        ax7.imshow(img[:, :, 7])
        plt.show()

