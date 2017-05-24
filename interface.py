import tkinter.messagebox as messagebox
from tkinter import *

from image_utils import *


class Application(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        Label(self, text="------观察数据集------").grid(row=0, column=0, columnspan=2)
        Button(self, text='遥感数据集说明', command=self.show_info).grid(row=1, column=0)
        Button(self, text='显示对应遥感图像', command=self.show_image).grid(row=1, column=1)
        Label(self, text="请输入遥感图像ID，默认为6100_2_2：").grid(row=2, column=0)
        self.image_id = Entry(self)
        self.image_id.grid(row=2, column=1)
        Label(self, text="请输入图像种类（A、M、P、RGB），默认为A：").grid(row=3, column=0)
        self.image_type = Entry(self)
        self.image_type.grid(row=3, column=1)

        Label(self, text="------数据处理------").grid(row=4, column=0, columnspan=2)
        Button(self, text='数据集处理说明', command=self.show_data_info).grid(row=1, column=0)

    def show_info(self):
        messagebox.showinfo('遥感数据集说明', '原始遥感图像数据集分为两种，一种是3波段的RGB图像，另一种是16'
                                       '波段的图像，由8个多光谱波段与8个短波红外（SWIR）波段组成，'
                                       '图像格式全为GeoTiff，采集自WorldView 3卫星。全色图的分辨率为'
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

    def show_data_info(self):
        messagebox.showinfo('数据集处理说明', '在观察完数据集之后，。')


app = Application()
# 设置窗口标题:
app.master.title('高分遥感影像分类')
# 主消息循环:
app.mainloop()
