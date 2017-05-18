import os
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from shapely.geometry import MultiPolygon, Polygon


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


# Create a mask from polygons
def mask_for_polygons(polygons, im_size):
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask


# Scale polygons to match image
def get_scalers(im_size, x_max, y_min):
    # they are flipped so that mask_for_polygons works correctly
    h, w = im_size
    w_ = w * (w / (w + 1))
    h_ = h * (h / (h + 1))
    return w_ / x_max, h_ / y_min


# Creating polygons from bit masks
def mask_to_polygons(mask, epsilon=5, min_area=1):
    # first, find contours with cv2: it's much faster than shapely
    image, contours, hierarchy = cv2.findContours(
        ((mask == 1) * 255).astype(np.uint8),
        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    # create approximate contours to have reasonable submission size
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True)
                       for cnt in contours]
    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons
