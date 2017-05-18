import csv
import sys

import numpy as np
import shapely.affinity
from keras.layers import Activation
from keras.layers import Flatten, Dense, Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from shapely import wkt
from sklearn.model_selection import train_test_split

from image_utils import M
from mask_utils import mask_for_polygons, get_scalers

csv.field_size_limit(sys.maxsize)

IM_ID = '6100_2_2'
POLY_TYPE = '5'  # Trees

# Load grid size
x_max = y_min = None
for _im_id, _x, _y in csv.reader(open('../data/grid_sizes.csv')):
    if _im_id == IM_ID:
        x_max, y_min = float(_x), float(_y)
        break

# Load train poly with shapely
train_polygons = dict()
for _im_id, _poly_type, _poly in csv.reader(open('../data/train_wkt_v4.csv')):
    if _im_id == IM_ID:
        train_polygons[_poly_type] = wkt.loads(_poly)
        break

# Read image
im_rgb = M(IM_ID)
im_size = im_rgb.shape[:2]


mask_map = list()
for key in train_polygons.keys():
    x_scaler, y_scaler = get_scalers(im_size, x_max, y_min)
    train_polygons_scaled = shapely.affinity.scale(train_polygons[key], xfact=x_scaler, yfact=y_scaler,
                                                   origin=(0, 0, 0))
    mask_map.append(mask_for_polygons(train_polygons_scaled, im_size))

mask = np.dstack(mask_map)

nrow, ncol = im_size
x = np.arange(0, nrow, 16)
y = np.arange(0, ncol, 16)

patch_list = list()
lbl_list = list()
for xstart, xend in zip(x[:-1], x[1:]):
    for ystart, yend in zip(y[:-1], y[1:]):
        patch_list.append(im_rgb[xstart:xend, ystart:yend, :])
        lbl_list.append(mask[xstart:xend, ystart:yend].mean())
patches = np.array(patch_list)
lbls = np.array(lbl_list)

xtrain, xtest, ytrain, ytest = train_test_split(patches, lbls, train_size=0.8, test_size=0.2)

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

model.fit(xtrain, ytrain, batch_size=32, epochs=10)

model.evaluate(xtrain, ytrain, batch_size=32)
model.evaluate(xtest, ytest, batch_size=32)


# pipeline = make_pipeline(StandardScaler(), SGDClassifier(loss='log'))
# pipeline.fit(xtrain, ytrain)
