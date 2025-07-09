import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog


class ImageApp:
    def __init__(self, root):
        # 创建主容器
        self.root = root
        self.root.title("图像处理工具")
        root.geometry("1600x900")  # 调整为更大的尺寸

        # 界面分为左右两部分
        self.left_frame = tk.Frame(self.root)
        self.right_frame = tk.Frame(self.root)

        # 设置布局，左右各占一半宽度
        self.left_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # 左侧Canvas显示原图
        self.left_canvas = tk.Canvas(self.left_frame)
        self.left_canvas.pack(expand=True, fill=tk.BOTH)

        # 右侧Canvas显示处理后的图像
        self.right_canvas = tk.Canvas(self.right_frame)
        self.right_canvas.pack(expand=True, fill=tk.BOTH)

        # 初始化变量
        self.original_image = None
        self.processed_image = None
        self.photo_original = None
        self.photo_processed = None

        # 创建加载图片的按钮，并绑定事件
        self.load_button = tk.Button(self.left_frame, text="加载图像", command=self.load_image)
        self.load_button.pack(side=tk.TOP, padx=5, pady=5)

        # 创建滑块用于调整阈值
        self.threshold_slider = tk.Scale(self.right_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                         label="二值化阈值", command=self.update_binarization)
        self.threshold_slider.pack(side=tk.TOP, padx=5, pady=5)

        # 设置默认阈值为128
        self.threshold = 128

        # 创建显示总灰度的标签
        self.gray_total_label = tk.Label(self.right_frame, text="总灰度: 0")
        self.gray_total_label.pack(side=tk.TOP, padx=5, pady=5)

        # 创建计算总灰度的按钮
        self.gray_total_button = tk.Button(self.right_frame, text="计算总灰度", command=self.calculate_gray_total)
        self.gray_total_button.pack(side=tk.TOP, padx=5, pady=5)

    def load_image(self):
        # 打开文件选择对话框，获取图片路径
        file_path = tk.filedialog.askopenfilename()
        if file_path:
            # 使用PIL读取图像
            self.original_image = Image.open(file_path)
            self.processed_image = self.original_image.copy()  # 创建处理后的图像副本

            # 显示原图在左侧Canvas上
            self.display_original_image()

            # 初始显示处理后图像（未调整阈值）
            self.display_processed_image()

    def display_original_image(self):
        if self.original_image:
            # 计算缩放比例，确保图片适应Canvas大小
            w, h = self.left_canvas.winfo_width(), self.left_canvas.winfo_height()
            img_w, img_h = self.original_image.size

            # 计算缩放因子
            scale = min(w / img_w, h / img_h)

            # 缩放图片
            resized = self.original_image.resize((int(img_w * scale), int(img_h * scale)), Image.ANTIALIAS)

            # 转换为PhotoImage并显示在Canvas上
            if self.photo_original is None:
                self.photo_original = ImageTk.PhotoImage(resized)
                self.left_canvas.create_image(0, 0, image=self.photo_original, anchor=tk.NW)
            else:
                # 避免重复对象，直接更新图片
                self.photo_original = ImageTk.PhotoImage(resized)
                self.left_canvas.itemconfig(self.left_canvas.find_all()[0], image=self.photo_original)

    def display_processed_image(self):
        if self.processed_image:
            # 计算缩放比例，确保图片适应Canvas大小
            w, h = self.right_canvas.winfo_width(), self.right_canvas.winfo_height()
            img_w, img_h = self.processed_image.size

            # 计算缩放因子
            scale = min(w / img_w, h / img_h)

            # 缩放图片
            resized = self.processed_image.resize((int(img_w * scale), int(img_h * scale)), Image.ANTIALIAS)

            # 转换为PhotoImage并显示在Canvas上
            if self.photo_processed is None:
                self.photo_processed = ImageTk.PhotoImage(resized)
                self.right_canvas.create_image(0, 0, image=self.photo_processed, anchor=tk.NW)
            else:
                # 避免重复对象，直接更新图片
                self.photo_processed = ImageTk.PhotoImage(resized)
                self.right_canvas.itemconfig(self.right_canvas.find_all()[0], image=self.photo_processed)

    def update_binarization(self, threshold):
        if not self.original_image:
            return

        # 转换为整数，确保阈值有效范围
        threshold = int(threshold)

        # 更新处理后的图像
        self.processed_image = self.binarize_image(self.original_image.copy(), threshold)

        # 显示更新后的处理后图像
        self.display_processed_image()

    def binarize_image(self, image, threshold):
        # 转换为灰度图像（如果需要）
        gray = image.convert("L")

        # 应用二值化处理
        processed = gray.point(lambda x: 255 if x > threshold else 0)

        return processed

    def calculate_gray_total(self):
        if not self.original_image:
            return

        # 转换为灰度图像
        gray_image = self.original_image.convert("L")

        # 计算总灰度值
        total_gray = sum(gray_image.getdata())

        # 更新标签显示总灰度
        self.gray_total_label.config(text=f"总灰度: {total_gray}")

    def on_resize(self, event):
        # 当窗口大小变化时，更新图片显示
        self.display_original_image()
        self.display_processed_image()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()