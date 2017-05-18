# This script shows the full training and prediction pipeline for a pixel-based classifier:
# we create a mask, train logistic regression on one-pixel patches, make prediction for all pixels,
# create and smooth polygons from pixels.
import csv
import sys

import numpy as np
import shapely.affinity
import tifffile as tiff
from shapely import wkt

from mask_utils import mask_for_polygons, get_scalers

csv.field_size_limit(sys.maxsize)

# We'll work on buildings (class 1) from image 6120_2_2. Fist load grid sizes and polygons.
IM_ID = '6120_2_2'
POLY_TYPE = '1'  # buildings

# Load grid size
x_max = y_min = None
for _im_id, _x, _y in csv.reader(open('../data/grid_sizes.csv')):
    if _im_id == IM_ID:
        x_max, y_min = float(_x), float(_y)
        break

# Load train poly with shapely
train_polygons = None
for _im_id, _poly_type, _poly in csv.reader(open('../data/train_wkt_v4.csv')):
    if _im_id == IM_ID and _poly_type == POLY_TYPE:
        train_polygons = wkt.loads(_poly)
        break

# Read image with tiff
im_rgb = tiff.imread('../data/three_band/{}.tif'.format(IM_ID)).transpose([1, 2, 0])
im_size = im_rgb.shape[:2]

x_scaler, y_scaler = get_scalers(im_size, x_max, y_min)

train_polygons_scaled = shapely.affinity.scale(
    train_polygons, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))

train_mask = mask_for_polygons(train_polygons_scaled, im_size)


# Now, let's train a very simple logistic regression classifier,
# just to get some noisy prediction to show how output mask is processed.
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score

xs = im_rgb.reshape(-1, 3).astype(np.float32)
ys = train_mask.reshape(-1)
pipeline = make_pipeline(StandardScaler(), SGDClassifier(loss='log'))

print('training...')
# do not care about overfitting here
pipeline.fit(xs, ys)
pred_ys = pipeline.predict_proba(xs)[:, 1]
print('average precision', average_precision_score(ys, pred_ys))
pred_mask = pred_ys.reshape(train_mask.shape)

