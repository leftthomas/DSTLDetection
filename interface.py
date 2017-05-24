from tkinter import *

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
image = np.zeros((m.shape[0], m.shape[1], 3))


class Application(Frame):
    def buttonListener1(self, event):
        # 测试不同波段组合显示的图像
        display_img(a)
        display_img(m)
        display_img(p)
        display_img(rgb)

    def buttonListener2(self, event):
        print('aa')

    def buttonListener3(self, event):
        print('aa')

    def buttonListener4(self, event):
        print('aa')

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.button1 = Button(self, text="显示遥感图片", width=10, height=5)
        self.button2 = Button(self, text="button2", width=10, height=5)
        self.button3 = Button(self, text="button3", width=10, height=5)
        self.button4 = Button(self, text="button4", width=10, height=5)

        self.button1.grid(row=0, column=0, padx=5, pady=5)
        self.button2.grid(row=0, column=1, padx=5, pady=5)
        self.button3.grid(row=1, column=0, padx=5, pady=5)
        self.button4.grid(row=1, column=1, padx=5, pady=5)

        self.button1.bind("<ButtonRelease-1>", self.buttonListener1)
        self.button2.bind("<ButtonRelease-1>", self.buttonListener2)
        self.button3.bind("<ButtonRelease-1>", self.buttonListener3)
        self.button4.bind("<ButtonRelease-1>", self.buttonListener4)


app = Application()
# 设置窗口标题:
app.master.title('高分遥感影像分类')
# 主消息循环:
app.mainloop()
