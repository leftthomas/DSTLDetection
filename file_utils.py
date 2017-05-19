import json

import pandas as pd
from shapely.geometry import Polygon

classType_to_filename = {
    1: '006_VEG_L2_SCRUBLAND',  # 灌木丛
    2: '006_VEG_L5_GROUP_TREES',  # 林地，群树
    3: '006_VEG_L5_STANDALONE_TREES',  # 单树
    4: '007_AGR_L2_CONTOUR_PLOUGHING_CROPLAND',  # 耕地，农田
    5: '007_AGR_L2_ORCHARD',  # 果园
    6: '007_AGR_L6_ROW_CROP'  # 农作物
}


# 读取指定的img对应的class的geojson文件的polygons
def load_geojson_to_polygons(img_id, class_type):
    file = json.load(open('../data/train_geojson_v3/' + img_id + '/' +
                          classType_to_filename.get(class_type) + '.geojson'))
    # print(file)
    polygon_list = list()
    for feature in file['features']:
        # print(len(feature['geometry']['coordinates'][0]))
        # print(Polygon(feature['geometry']['coordinates'][0]))
        polygon_list.append(Polygon(feature['geometry']['coordinates'][0]))
    # print(polygon_list)
    return polygon_list

# load_geojson_to_polygons('6100_2_2', 6)


# 从grid_sizes.csv中读取指定图像的 Xmax 和 Ymin
def get_xmax_ymin(image_id):
    gs = pd.read_csv('../data/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
    xmax, ymin = gs[gs.ImageId == image_id].iloc[0, 1:].astype(float)
    return xmax, ymin
