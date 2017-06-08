import random

import matplotlib

matplotlib.use("TkAgg")
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from image_utils import stretch_n, M
from mask_utils import generate_mask_for_image_and_class, display_predict_result
from network import get_unet, calc_jacc

class_number = 10
DF = pd.read_csv('../data/train_wkt_v4.csv')
GS = pd.read_csv('../data/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
ISZ = 160


def stick_all_train():
    print("构造训练集，保存至本地")
    # 因为M段图有些图尺寸不一样，这里统一取835*835，基本上都是837*849，接近1：1
    s = 835

    x = np.zeros((5 * s, 5 * s, 8))
    y = np.zeros((5 * s, 5 * s, class_number))

    ids = sorted(DF.ImageId.unique())
    # print('训练集遥感图像总张数：',len(ids))
    for i in range(5):
        for j in range(5):
            id = ids[5 * i + j]

            img = M(id)
            img = stretch_n(img)
            print(id, ':', img.shape)
            x[s * i:s * i + s, s * j:s * j + s, :] = img[:s, :s, :]
            for z in range(class_number):
                y[s * i:s * i + s, s * j:s * j + s, z] = generate_mask_for_image_and_class(
                    (img.shape[0], img.shape[1]), id, z + 1)[:s, :s]
    # 看下mask有没有做错，数据是不是为[0,1]
    # print(np.amax(y), np.amin(y))

    np.save('data/x_trn_%d' % class_number, x)
    np.save('data/y_trn_%d' % class_number, y)


# 生成patches
def get_patches(img, msk, amt=10000, aug=True):
    xm, ym = img.shape[0] - ISZ, img.shape[1] - ISZ

    x, y = [], []
    # 每一类的阈值
    tr = [0.4, 0.1, 0.1, 0.15, 0.3, 0.95, 0.1, 0.05, 0.001, 0.005]
    for i in range(amt):
        xc = random.randint(0, xm)
        yc = random.randint(0, ym)

        im = img[xc:xc + ISZ, yc:yc + ISZ]
        ms = msk[xc:xc + ISZ, yc:yc + ISZ]

        for j in range(class_number):
            sm = np.sum(ms[:, :, j])
            if sm / ISZ ** 2 > tr[j]:
                # 做一些变换
                if aug:
                    if random.uniform(0, 1) > 0.5:
                        # 垂直翻转
                        im = im[::-1]
                        ms = ms[::-1]
                    if random.uniform(0, 1) > 0.5:
                        # 水平翻转
                        im = im[:, ::-1]
                        ms = ms[:, ::-1]

                x.append(im)
                y.append(ms)
    # x需要转换到[-1, 1]区间，归一化，方便计算，同时需要转置，即将(n,160,160,8)变为(n,8,160,160)
    # (0, 3, 1, 2)即根据原shape索引(n[0],8[3],160[1],160[2])转变shape
    x, y = 2 * np.transpose(x, (0, 3, 1, 2)) - 1, np.transpose(y, (0, 3, 1, 2))
    # print(x.shape, y.shape)
    return x, y


def make_val():
    print("构造验证集")
    img = np.load('data/x_trn_%d.npy' % class_number)
    msk = np.load('data/y_trn_%d.npy' % class_number)
    x, y = get_patches(img, msk, amt=3000)

    np.save('data/x_tmp_%d' % class_number, x)
    np.save('data/y_tmp_%d' % class_number, y)


def train_net():
    print("start train net")
    x_val, y_val = np.load('data/x_tmp_%d.npy' % class_number), np.load('data/y_tmp_%d.npy' % class_number)
    img, msk = np.load('data/x_trn_%d.npy' % class_number), np.load('data/y_trn_%d.npy' % class_number)
    x_trn, y_trn = get_patches(img, msk)

    model = get_unet()
    model.load_weights('weights/unet_10_jk0.7878')
    # 回调函数，在epoch结束后保存在验证集上性能最好的模型
    model_checkpoint = ModelCheckpoint('weights/unet_tmp.hdf5', monitor='loss', save_best_only=True)
    model.fit(x_trn, y_trn, batch_size=64, epochs=1, verbose=1, shuffle=True,
              callbacks=[model_checkpoint], validation_data=(x_val, y_val))
    score, trs = calc_jacc(model)
    print('val jk', score)
    model.save_weights('weights/unet_10_jk%.4f' % score)

    return model


def predict_id(id, model, trs):
    img = M(id)
    x = stretch_n(img)
    # 因为M段图基本都是（837，848，8）这样的尺寸，但神经网络需要的是(8,160,160)
    # 所以满足这要求的最小的尺寸就是960(160*6)，刚好分6次
    cnv = np.zeros((960, 960, 8)).astype(np.float32)
    prd = np.zeros((class_number, 960, 960)).astype(np.float32)
    cnv[:img.shape[0], :img.shape[1], :] = x

    for i in range(0, 6):
        line = []
        for j in range(0, 6):
            # 取(160,160,8)的图像块出来
            line.append(cnv[i * ISZ:(i + 1) * ISZ, j * ISZ:(j + 1) * ISZ])
        # 转换到[-1, 1]区间，归一化，方便计算，同时需要转置，即将(n,160,160,8)变为(n,8,160,160)
        x = 2 * np.transpose(line, (0, 3, 1, 2)) - 1
        tmp = model.predict(x, batch_size=4)
        for j in range(tmp.shape[0]):
            prd[:, i * ISZ:(i + 1) * ISZ, j * ISZ:(j + 1) * ISZ] = tmp[j]
    # 转成[0,1]的mask
    for i in range(class_number):
        prd[i] = prd[i] > trs[i]
    # 记得将图像尺寸转回原尺寸
    return prd[:, :img.shape[0], :img.shape[1]]


def check_predict(id='6100_3_2'):
    model = get_unet()
    model.load_weights('weights/unet_10_jk0.7914')
    # 这个阈值是calc_jacc时算出来的最佳阈值
    msk = predict_id(id, model, [0.4, 0.1, 0.4, 0.3, 0.3, 0.5, 0.3, 0.6, 0.1, 0.1])
    img = M(id)
    display_predict_result(img, msk)

# stick_all_train()
# make_val()
# model = train_net()
# check_predict()
