from file_utils import *
from image_utils import *
from mask_utils import *

# 测试A、M、P、RGB四个波段图像shape
imageId = '6100_2_2'
class_type = 1
a = A(imageId)
m = M(imageId)
p = P(imageId)
rgb = RGB(imageId)
print('A-shape:', a.shape, 'dtype:', a.dtype, 'M-shape:', m.shape, 'dtype:', m.dtype, 'P-shape:',
      p.shape, 'dtype:', p.dtype, 'RGB-shape:', rgb.shape, 'dtype:', rgb.dtype)
# 测试不同波段组合显示的图像
display_img(a)
display_img(m)
display_img(p)
display_img(rgb)
image = np.zeros((m.shape[0], m.shape[1], 3))
# 从M段中取出RGB波段组合成图像
image[:, :, 0] = m[:, :, 4]  # red
image[:, :, 1] = m[:, :, 2]  # green
image[:, :, 2] = m[:, :, 1]  # blue
# 对比原图与做过对比度加强的图像，原图其实与RGB段的图是一样的
display_img(image)
# 测试线性拉伸
image = stretch_n(image)
display_img(image)
# 测试从geojson文件中读取polygons
polygons_geojson = load_geojson_to_polygons(imageId, class_type)
# 测试从csv文件中读取polygons
polygons_csv = load_wkt_to_polygons(imageId, class_type)
# 测试从grid_sizes.csv中读取指定图像的 Xmax 和 Ymin
x_max, y_min = get_xmax_ymin(imageId)
# 测试得到polygons地理坐标到像素坐标点映射的缩放因子
x_scale, y_scale = get_scales((image.shape[0], image.shape[1]), x_max, y_min)
# 测试将polygons转化成mask
mask = polygons_to_mask(polygons_geojson, (image.shape[0], image.shape[1]), False, x_scale, y_scale)
# 测试polys的绘制
display_polygons(polygons_geojson, image, x_scale, y_scale)
# 测试mask的绘制
display_img(mask)
