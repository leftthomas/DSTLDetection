import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import wkt, affinity

from utils import stretch_n, RGB, M

# Select a class and image id to be drawn
Class = 1
ImageID = '6120_2_2'


def truth_polys(image_id, class_id, W, H):
    x = pd.read_csv('../data/train_wkt_v4.csv')
    rows = x.loc[(x.ImageId == image_id) & (x.ClassType == class_id), 'MultipolygonWKT']
    mp = wkt.loads(rows.values[0])
    grid_sizes = pd.read_csv('../data/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
    xmax, ymin = grid_sizes[grid_sizes.ImageId == ImageID].iloc[0, 1:].astype(float)
    W_ = W * (W / (W + 1.))
    H_ = H * (H / (H + 1.))
    x_scaler = W_ / xmax
    y_scaler = H_ / ymin
    return affinity.scale(mp, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))

# Read threeband image
rgb = RGB(ImageID)

# Read 16band m image
img_m = M(ImageID)
img_m = cv2.resize(img_m, tuple(reversed(rgb.shape[:2])))

# Turn m image into rgb color
x = np.zeros_like(rgb)
x[:, :, 0] = img_m[:, :, 4]
x[:, :, 1] = img_m[:, :, 2]
x[:, :, 2] = img_m[:, :, 1]
x = stretch_n(x).copy()

H = len(x)
W = len(x[0])
# Read Polygons
polys = truth_polys(ImageID, Class, W, H)

# Add polygons to the x image --Edit: 05.25.17  included hole verteces in polygons
int_vertices = lambda x: np.array(x).round().astype(np.int32)
for poly_id, poly in enumerate(polys):
    x1, y1, x2, y2 = [int(pb) for pb in poly.bounds]
    xys = int_vertices(poly.exterior.coords)
    cv2.polylines(x, [xys], True, (255, 0, 0), 3)
    for pi in poly.interiors:
        ixys = int_vertices(pi.coords)
        cv2.polylines(x, [ixys], True, (255, 0, 0), 3)


# Plot
fig, ax = plt.subplots(figsize=(9, 9))
ax.imshow(x)
plt.show()
