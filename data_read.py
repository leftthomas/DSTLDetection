import tifffile as tiff
from shapely import wkt

img_filename_16bandA = '../data/sixteen_band/6100_1_3_A.tif'
img_filename_16bandM = '../data/sixteen_band/6100_1_3_M.tif'
img_filename_16bandP = '../data/sixteen_band/6100_1_3_P.tif'

A = tiff.imread(img_filename_16bandA)
M = tiff.imread(img_filename_16bandM)
P = tiff.imread(img_filename_16bandP)
print(A.shape, M.shape, P.shape)

wkt.loads()
# tiff.imshow(P)
