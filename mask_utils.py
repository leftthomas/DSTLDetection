import cv2
import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from shapely import affinity
from file_utils import get_xmax_ymin, get_scales, load_wkt_to_polygons
from image_utils import stretch_n


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


def generate_mask_for_image_and_class(raster_size, image_id, class_type):
    x_max, y_min = get_xmax_ymin(image_id)
    xf, yf = get_scales(raster_size, x_max, y_min)
    polygon_list = load_wkt_to_polygons(image_id, class_type)
    mask = polygons_to_mask(polygon_list, raster_size, False, xf, yf)
    return mask


# 根据load_all_geojson或load_all_wkt得到的dict polygons读取所有masks
def load_all_masks(polygons, im_size, x_scale, y_scale):
    masks = list()
    for key in polygons.keys():
        masks.append(polygons_to_mask(polygons[key], im_size, False, x_scale, y_scale))
    return masks


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


# 显示所有mask
def display_all_mask(masks):
    fig = plt.figure()
    ax = fig.add_subplot(3, 2, 1)
    ax.imshow(masks[0])
    ax1 = fig.add_subplot(3, 2, 2)
    ax1.imshow(masks[1])
    ax2 = fig.add_subplot(3, 2, 3)
    ax2.imshow(masks[2])
    ax3 = fig.add_subplot(3, 2, 4)
    ax3.imshow(masks[3])
    ax4 = fig.add_subplot(3, 2, 5)
    ax4.imshow(masks[4])
    ax5 = fig.add_subplot(3, 2, 6)
    ax5.imshow(masks[5])
    plt.show()


def display_predict_result(img, masks):
    image = np.zeros((img.shape[0], img.shape[1], 3))
    # 从M段中取出RGB波段组合成图像
    image[:, :, 0] = img[:, :, 4]  # red
    image[:, :, 1] = img[:, :, 2]  # green
    image[:, :, 2] = img[:, :, 1]  # blue
    stretch_image = stretch_n(image)

    fig = plt.figure()
    ax1 = fig.add_subplot(3, 4, 1)
    ax1.set_title('Original')
    ax1.imshow(image)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2 = fig.add_subplot(3, 4, 2)
    ax2.set_title('Stretched')
    ax2.imshow(stretch_image)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3 = fig.add_subplot(3, 4, 3)
    ax3.set_title('Buildings')
    ax3.imshow(masks[0])
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax4 = fig.add_subplot(3, 4, 4)
    ax4.set_title('Misc')
    ax4.imshow(masks[1])
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax5 = fig.add_subplot(3, 4, 5)
    ax5.set_title('Road')
    ax5.imshow(masks[2])
    ax5.set_xticks([])
    ax5.set_yticks([])
    ax6 = fig.add_subplot(3, 4, 6)
    ax6.set_title('Track')
    ax6.imshow(masks[3])
    ax6.set_xticks([])
    ax6.set_yticks([])
    ax7 = fig.add_subplot(3, 4, 7)
    ax7.set_title('Trees')
    ax7.imshow(masks[4])
    ax7.set_xticks([])
    ax7.set_yticks([])
    ax8 = fig.add_subplot(3, 4, 8)
    ax8.set_title('Crops')
    ax8.imshow(masks[5])
    ax8.set_xticks([])
    ax8.set_yticks([])
    ax9 = fig.add_subplot(3, 4, 9)
    ax9.set_title('Waterway')
    ax9.imshow(masks[6])
    ax9.set_xticks([])
    ax9.set_yticks([])
    ax10 = fig.add_subplot(3, 4, 10)
    ax10.set_title('Standing water')
    ax10.imshow(masks[7])
    ax10.set_xticks([])
    ax10.set_yticks([])
    ax11 = fig.add_subplot(3, 4, 11)
    ax11.set_title('Vehicle Large')
    ax11.imshow(masks[8])
    ax11.set_xticks([])
    ax11.set_yticks([])
    ax12 = fig.add_subplot(3, 4, 12)
    ax12.set_title('Vehicle Small')
    ax12.imshow(masks[9])
    ax12.set_xticks([])
    ax12.set_yticks([])
    plt.show()
