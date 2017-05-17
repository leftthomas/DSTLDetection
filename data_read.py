import tifffile as tiff
from matplotlib import pyplot

img_filename_16bandA = '../data/sixteen_band/6100_1_3_A.tif'
img_filename_16bandM = '../data/sixteen_band/6100_1_3_M.tif'
img_filename_16bandP = '../data/sixteen_band/6100_1_3_P.tif'

P = tiff.imread(img_filename_16bandP)
tiff.imshow(P)
pyplot.show()
