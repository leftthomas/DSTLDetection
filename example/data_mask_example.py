import csv
import sys

import cv2
import numpy as np
import shapely.affinity
import tifffile as tiff

csv.field_size_limit(sys.maxsize)

IM_ID = '6120_2_2'
# buildings
POLY_TYPE = '1'

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
        train_polygons = shapely.wkt.loads(_poly)
        break

# Read image with tiff
im_rgb = tiff.imread('../data/three_band/{}.tif'.format(IM_ID)).transpose([1, 2, 0])
im_size = im_rgb.shape[:2]


def get_scalers():
    # they are flipped so that mask_for_polygons works correctly
    h, w = im_size
    w_ = w * (w / (w + 1))
    h_ = h * (h / (h + 1))
    return w_ / x_max, h_ / y_min


x_scaler, y_scaler = get_scalers()

train_polygons_scaled = shapely.affinity.scale(
    train_polygons, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))


def mask_for_polygons(polygons):
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


train_mask = mask_for_polygons(train_polygons_scaled)

print(train_mask.shape)
im_rgb = tiff.imread('../data/three_band/{}.tif'.format(IM_ID)).transpose([1, 2, 0])
print(im_rgb.shape)
