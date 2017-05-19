from collections import defaultdict

import cv2
import numpy as np
from matplotlib import pyplot as plt
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon

from file_utils import get_xmax_ymin, load_geojson_to_polygons, get_scales


# 将polygons转化成mask,注意,已经转化过的polygons不需要再做转化了
def polygons_to_mask(polygons, im_size, has_transformed=True, x_scale=0, y_scale=0):
    if not has_transformed:
        polygons = affinity.scale(polygons, xfact=x_scale, yfact=y_scale, origin=(0, 0, 0))
    img_mask = np.zeros(im_size, np.uint8)
    if polygons is None:
        return img_mask
    coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [coords(poly.exterior.coords) for poly in polygons]
    # print(exteriors[0])
    interiors = [coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    # print(interiors)
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask


# 将mask转化到polygons
def mask_to_polygons(mask, epsilon=5, min_area=1):
    # first, find contours with cv2: it's much faster than shapely
    image, contours, hierarchy = cv2.findContours(
        ((mask == 1) * 255).astype(np.uint8),
        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    # create approximate contours to have reasonable submission size
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True)
                       for cnt in contours]
    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons


def generate_mask_for_image_and_class(raster_size, image_id, class_type):
    x_max, y_min = get_xmax_ymin(image_id)
    xf, yf = get_scales(raster_size, x_max, y_min)
    polygon_list = load_geojson_to_polygons(image_id, class_type)
    mask = polygons_to_mask(polygon_list, raster_size, False, xf, yf)
    return mask


def display_polygons(polygons, img, x_scale, y_scale):
    polygons = affinity.scale(polygons, xfact=x_scale, yfact=y_scale, origin=(0, 0, 0))
    vertices = lambda x: np.array(x).round().astype(np.int32)
    for poly_id, poly in enumerate(polygons):
        xys = vertices(poly.exterior.coords)
        cv2.polylines(img, [xys], True, (255, 0, 0), 2)
        for pi in poly.interiors:
            i_xys = vertices(pi.coords)
            cv2.polylines(img, [i_xys], True, (0, 255, 0), 2)
    plt.imshow(img)
    plt.show()
