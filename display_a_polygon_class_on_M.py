"""
This Scritp Show Traning Polygons on Satalite İmages for each Class
"""

import os

# Import Libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from shapely import wkt, affinity

import utils

"""
Classes 
        'Buildings'        :1,
        'Structures '      :2,
        'Road'             :3,
        'Track'            :4,
        'Trees'            :5,
        'Crops'            :6,
        'Waterway'         :7,
        'StandingWater'    :8,
        'VehicleLarge'     :9,
        'VehicleSmall'     :10,
"""

# Select a class and image id to be drawn
Class = 4
ImageID = '6120_2_2'  # '6120_2_2', '6100_1_3', '6140_3_1','6110_3_1','6100_2_3','6140_1_2','6120_2_0','6100_2_2','6110_1_2','6070_2_3','6110_4_0','6090_2_0','6060_2_3'


# ---------------------------------------

def adjust_contrast(x):
    for i in range(3):
        x[:, :, i] = utils.stretch_n(x[:, :, i])
    return x.astype(np.uint8)


def truth_polys(image_id, class_id, W, H):
    x = pd.read_csv('../../data/train_wkt_v4.csv')
    rows = x.loc[(x.ImageId == image_id) & (x.ClassType == class_id), 'MultipolygonWKT']
    mp = wkt.loads(rows.values[0])
    grid_sizes = pd.read_csv('../../data/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
    xmax, ymin = grid_sizes[grid_sizes.ImageId == ImageID].iloc[0, 1:].astype(float)
    W_ = W * (W / (W + 1.))
    H_ = H * (H / (H + 1.))
    x_scaler = W_ / xmax
    y_scaler = H_ / ymin
    return affinity.scale(mp, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))


# -----------------------------------------------------------------------------------------


# Read threeband image
rgbfile = os.path.join('/Users/left/workspace/data/three_band', '{}.tif'.format(ImageID))
rgb = tifffile.imread(rgbfile)
rgb = np.rollaxis(rgb, 0, 3)

# Read 16band m image
mfile = os.path.join('/Users/left/workspace/data/sixteen_band', '{}_M.tif'.format(ImageID))
img_m = tifffile.imread(mfile)
img_m = np.rollaxis(img_m, 0, 3)
img_m = cv2.resize(img_m, tuple(reversed(rgb.shape[:2])))

# Turn m image into rgb color
x = np.zeros_like(rgb)
x[:, :, 0] = img_m[:, :, 4]
x[:, :, 1] = img_m[:, :, 2]
x[:, :, 2] = img_m[:, :, 1]
x = adjust_contrast(x).copy()

H = len(x)
W = len(x[0])
# Read Polygons
polys = truth_polys(ImageID, Class, W, H)

# Add polygons to the x image --Edit: 05.25.17  included hole verteces in polygons
# patches=[]
int_vertices = lambda x: np.array(x).round().astype(np.int32)
for poly_id, poly in enumerate(polys):
    # x1,y1,x2,y2 = [int(pb) for pb in poly.bounds]
    xys = int_vertices(poly.exterior.coords)
    cv2.polylines(x, [xys], True, (255, 0, 0), 3)
    for pi in poly.interiors:
        ixys = int_vertices(pi.coords)
        cv2.polylines(x, [ixys], True, (255, 0, 0), 3)
        # patches.append(np.hstack([x[y1-PADDING:y2+PADDING, x1-PADDING:x2+PADDING,:]]))

# # To focus on each element
# PADDING = 10
# W = 3396
# H = 3348
# patches = []
# titles = []
# for poly_id, poly in enumerate(polys):
#    x1,y1,x2,y2 = [int(pb) for pb in poly.bounds]
#    cv2.rectangle(x, (x1,y1), (x2,y2), (255,0,0), 1)
#    cv2.rectangle(img_p, (x1,y1), (x2,y2), (255,0,0), 1)
#    patches.append(np.hstack([x[y1-PADDING:y2+PADDING, x1-PADDING:x2+PADDING,:], img_p[y1-PADDING:y2+PADDING, x1-PADDING:x2+PADDING,:]]))
#    titles.append("ImageID: {} -- poly_id: {}".format(ImageID, poly_id))
#


# -----------------------------------------------------------------------------------------
# Plot
fig, ax = plt.subplots(figsize=(9, 9))
ax.imshow(x)
plt.savefig(ImageID + ".png")

# fig, ax = plt.subplots(1, 1, figsize=(10,10))
# for i in range(10):
#    ax.imshow(patches[i])
