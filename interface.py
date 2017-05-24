import tkinter.messagebox as messagebox
from tkinter import *

from baseline import check_predict
from file_utils import *
from image_utils import *
from mask_utils import *


class Application(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        Label(self, text="----------------------------------观察数据集----------------------------------").grid(row=0,
                                                                                                           column=0,
                                                                                                           columnspan=2)
        Button(self, text='遥感数据集说明', command=self.show_info).grid(row=1, column=0)
        Button(self, text='显示对应遥感图像', command=self.show_image).grid(row=1, column=1)
        Label(self, text="遥感图像ID，默认为6100_2_2：").grid(row=2, column=0)
        self.image_id = Entry(self)
        self.image_id.grid(row=2, column=1)
        Label(self, text="图像种类（A、M、P、RGB），默认为A：").grid(row=3, column=0)
        self.image_type = Entry(self)
        self.image_type.grid(row=3, column=1)

        Label(self, text="----------------------------------数据集处理----------------------------------").grid(row=4,
                                                                                                           column=0,
                                                                                                           columnspan=2)
        Button(self, text='训练集选择说明', command=self.show_train_info).grid(row=5, column=0)
        Button(self, text='数据集处理说明', command=self.show_data_info).grid(row=5, column=1)
        Button(self, text='显示处理后的图像（M8波段）', command=self.show_handled_m_image).grid(row=6, column=0)
        Button(self, text='显示处理后的图像（提取RGB波段）', command=self.show_handled_rgb_image).grid(row=6, column=1)

        Label(self, text="----------------------------------mask生成----------------------------------").grid(row=7,
                                                                                                            column=0,
                                                                                                            columnspan=2)
        Button(self, text='mask生成说明', command=self.show_mask_info).grid(row=8, column=0)
        Button(self, text='植被种类说明', command=self.show_vegs_info).grid(row=8, column=1)
        Label(self, text="植被种类，默认为2：").grid(row=9, column=0)
        self.image_class = Entry(self)
        self.image_class.grid(row=9, column=1)
        Button(self, text='显示多边形区域', command=self.show_polygons).grid(row=10, column=0)
        Button(self, text='显示mask', command=self.show_mask).grid(row=10, column=1)

        Label(self, text="----------------------------------神经网络训练----------------------------------").grid(row=11,
                                                                                                            column=0,
                                                                                                            columnspan=2)
        Button(self, text='神经网络训练说明', command=self.show_net_info).grid(row=12, column=0)
        Button(self, text='开始训练', command=self.train_net).grid(row=12, column=1)

        Label(self, text="----------------------------------结果验证----------------------------------").grid(row=13,
                                                                                                          column=0,
                                                                                                          columnspan=2)
        Label(self, text="待预测遥感图像ID，默认为6100_3_2：").grid(row=14, column=0)
        self.image_target = Entry(self)
        self.image_target.grid(row=14, column=1)
        Button(self, text='生成预测的各类别mask', command=self.predict).grid(row=15, column=0, columnspan=2)

    def show_info(self):
        messagebox.showinfo('遥感数据集说明', '原始遥感图像数据集分为两种，一种是3波段的RGB图像，另一种是16'
                                       '波段的图像，由8个多光谱（M）波段与8个短波红外（A）波段组成，'
                                       '图像格式全为GeoTiff，采集自WorldView 3卫星。全色（P）图的分辨率为'
                                       '0.31米，多光谱波段图的分辨率为1.24米，短波红外波段图分辨率'
                                       '为7.5米。总共图像为450张（46.05GB），训练集为25张。')

    def show_image(self):
        img_id = self.image_id.get() or '6100_2_2'
        img_type = self.image_type.get() or 'A'
        # 显示不同波段组合的图像
        try:
            if img_type == 'A':
                a = A(img_id)
                display_img(a)
            elif img_type == 'M':
                m = M(img_id)
                display_img(m)
            elif img_type == 'P':
                p = P(img_id)
                display_img(p)
            elif img_type == 'RGB':
                rgb = RGB(img_id)
                display_img(rgb)
            else:
                messagebox.showinfo('提示', '输入图像种类有误')
        except:
            messagebox.showinfo('提示', '输入图像ID有误')

    def show_train_info(self):
        messagebox.showinfo('训练集选择说明', '在观察完数据集之后，发现P只有一个波段，而A的分辨率相对M'
                                       '又太小，且M中包含RGB波段，所以确定采用M作为训练集图像。')

    def show_data_info(self):
        messagebox.showinfo('数据集处理说明', '因为M数据位数是16位,属于uint16类型，需要将其转为float32'
                                       '类型，使用5%的线性拉伸，方便后续计算和观察。')

    def show_handled_rgb_image(self):
        img_id = self.image_id.get() or '6100_2_2'
        try:
            m = M(img_id)
            image = np.zeros((m.shape[0], m.shape[1], 3))
            # 从M段中取出RGB波段组合成图像
            image[:, :, 0] = m[:, :, 4]  # red
            image[:, :, 1] = m[:, :, 2]  # green
            image[:, :, 2] = m[:, :, 1]  # blue
            # 线性拉伸
            image = stretch_n(image)
            display_img(image)
        except:
            messagebox.showinfo('提示', '输入图像ID有误')

    def show_handled_m_image(self):
        img_id = self.image_id.get() or '6100_2_2'
        try:
            m = M(img_id)
            # 线性拉伸
            image = stretch_n(m)
            display_img(image)
        except:
            messagebox.showinfo('提示', '输入图像ID有误')

    def show_mask_info(self):
        messagebox.showinfo('mask生成说明', '因为数据集的训练数据是以多边形区域（MultiPolygon）的形式提供的，所以需要'
                                        '先将其转为mask再输入神经网络中进行训练。')

    def show_vegs_info(self):
        messagebox.showinfo('植被种类说明', '1:灌木丛；2:林地；3:单树；4:耕地；5:果园；6:农作物。')

    def show_polygons(self):
        img_id = self.image_id.get() or '6100_2_2'
        img_class = self.image_class.get() or '2'
        try:
            m = M(img_id)
            image = np.zeros((m.shape[0], m.shape[1], 3))
            image[:, :, 0] = m[:, :, 4]
            image[:, :, 1] = m[:, :, 2]
            image[:, :, 2] = m[:, :, 1]
            image = stretch_n(image)
            if img_class in ['1', '2', '3', '4', '5', '6']:
                # 从geojson文件中读取polygons
                polygons_geojson = load_geojson_to_polygons(img_id, int(img_class))
                # 从grid_sizes.csv中读取指定图像的 Xmax 和 Ymin
                x_max, y_min = get_xmax_ymin(img_id)
                # 得到polygons地理坐标到像素坐标点映射的缩放因子
                x_scale, y_scale = get_scales((image.shape[0], image.shape[1]), x_max, y_min)
                # polys的绘制
                display_polygons(polygons_geojson, image, x_scale, y_scale)
            else:
                messagebox.showinfo('提示', '输入植被种类有误')
        except:
            messagebox.showinfo('提示', '输入图像ID有误')

    def show_mask(self):
        img_id = self.image_id.get() or '6100_2_2'
        img_class = self.image_class.get() or '2'
        try:
            m = M(img_id)
            image = np.zeros((m.shape[0], m.shape[1], 3))
            image[:, :, 0] = m[:, :, 4]
            image[:, :, 1] = m[:, :, 2]
            image[:, :, 2] = m[:, :, 1]
            image = stretch_n(image)
            if img_class in ['1', '2', '3', '4', '5', '6']:
                # 从geojson文件中读取polygons
                polygons_geojson = load_geojson_to_polygons(img_id, int(img_class))
                # 从grid_sizes.csv中读取指定图像的 Xmax 和 Ymin
                x_max, y_min = get_xmax_ymin(img_id)
                # 得到polygons地理坐标到像素坐标点映射的缩放因子
                x_scale, y_scale = get_scales((image.shape[0], image.shape[1]), x_max, y_min)
                # 将polygons转化成mask
                mask = polygons_to_mask(polygons_geojson, (image.shape[0], image.shape[1]), False, x_scale, y_scale)
                # 测试mask的绘制
                display_img(mask)
            else:
                messagebox.showinfo('提示', '输入植被种类有误')
        except:
            messagebox.showinfo('提示', '输入图像ID有误')

    def show_net_info(self):
        messagebox.showinfo('神经网络训练说明', '采用的神经网络结构为U-Net，在预训练的权值基础上进行'
                                        'fine-tuning，batch_size设为64，epochs设为1，optimizer'
                                        '设为Adam，损失函数设为binary_crossentropy。')

    def train_net(self):
        print('train')
        # stick_all_train()
        # model = train_net()

    def predict(self):
        img_target = self.image_target.get() or '6100_3_2'
        try:
            check_predict(img_target)
        except:
            messagebox.showinfo('提示', '输入图像ID有误')


app = Application()
# 设置窗口标题:
app.master.title('高分遥感影像分类')
# 主消息循环:
app.mainloop()
