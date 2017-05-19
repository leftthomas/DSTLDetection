import numpy as np
from keras.layers import Activation
from keras.layers import Flatten, Dense, Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from sklearn.model_selection import train_test_split

from file_utils import get_xmax_ymin, load_all_geojson
from image_utils import M
from mask_utils import get_scales, load_all_masks

IM_ID = '6100_2_2'
# Read image
im_rgb = M(IM_ID)
im_size = im_rgb.shape[:2]
# print(im_size)
x_max, y_min = get_xmax_ymin(IM_ID)
x_scale, y_scale = get_scales(im_size, x_max, y_min)
# 载入所有polygons
train_polygons = load_all_geojson(IM_ID)
masks = np.dstack(load_all_masks(train_polygons, im_size, x_scale, y_scale))
# print(len(masks))

# 构建训练集，本质是在图上取16*16的patchs
n_row, n_col = im_size
x = np.arange(0, n_row, 16)
y = np.arange(0, n_col, 16)
# print(x,y)
patch_list = list()
lbl_list = list()
for x_start, x_end in zip(x[:-1], x[1:]):
    for y_start, y_end in zip(y[:-1], y[1:]):
        patch_list.append(im_rgb[x_start:x_end, y_start:y_end, :])
        # 把此patch的mask的mean值作为对应的分类结果
        lbl_list.append(masks[x_start:x_end, y_start:y_end].mean())
patches = np.array(patch_list)
lbls = np.array(lbl_list)
# print(patches.shape,lbls.shape)

x_train, x_test, y_train, y_test = train_test_split(patches, lbls, train_size=0.8, test_size=0.2)

model = Sequential()
model.add(Conv2D(64, (1, 1), input_shape=(16, 16, 8), padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(64, (1, 1), padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D((2, 2)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=10)


# pipeline = make_pipeline(StandardScaler(), SGDClassifier(loss='log'))
# pipeline.fit(x_train, y_train)
