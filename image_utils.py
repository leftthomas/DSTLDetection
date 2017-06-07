import os

# 解决matplotlib与tkinter冲突的问题
import matplotlib

matplotlib.use("TkAgg")
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
        # 这里直接拉伸到[0,1]之间，不需要先拉伸到[0,255]后面再转
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


# 从M段提取出RGB并进行线性拉伸，保存训练图片或全部图片(主要是方便观察图片本来长什么样)
def generate_images_from_m(is_train=True):
    if not is_train:
        # 只提取ID
        img_names = [os.path.splitext(img)[0] for img in os.listdir('../data/three_band')]
    else:
        img_names = os.listdir('../data/train_geojson_v3')
    # print(img_names)
    # 记得要把Mac下的.DS_Store隐藏文件去掉
    if img_names.__contains__('.DS_Store'):
        img_names.remove('.DS_Store')
        # print(img_names)

    for img_name in img_names:
        m = M(img_name)
        image = np.zeros((m.shape[0], m.shape[1], 3))
        image[:, :, 0] = m[:, :, 4]  # red
        image[:, :, 1] = m[:, :, 2]  # green
        image[:, :, 2] = m[:, :, 1]  # blue
        image = stretch_n(image)
        if not is_train:
            plt.imsave('all_images/' + img_name + '.png', image)
        else:
            plt.imsave('train_images/' + img_name + '.png', image)


# generate_images_from_m()
# generate_images_from_m(False)
