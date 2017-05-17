import csv
import sys

import pandas as pd
import tifffile as tiff

csv.field_size_limit(sys.maxsize)

df = pd.read_csv('../data/train_wkt_v4.csv')

img_filename_16bandA = '../data/sixteen_band/6100_1_3_A.tif'
img_filename_16bandM = '../data/sixteen_band/6100_1_3_M.tif'
img_filename_16bandP = '../data/sixteen_band/6100_1_3_P.tif'

A = tiff.imread(img_filename_16bandA)
M = tiff.imread(img_filename_16bandM)
P = tiff.imread(img_filename_16bandP)
print(A.shape, M.shape, P.shape)
# tiff.imshow(P)
