import cv2
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
import os
import tkinter
from tkinter import filedialog


def calculate_mse_psnr_ssim(img1, img2, max_pixel=255):
    """
    计算两幅图像的MSE、PSNR和SSIM，兼容灰度图和彩色图。

    参数:
        img1, img2: 输入图像，可以是灰度图或彩色图（numpy数组）。
        max_pixel: 图像的最大像素值（默认255，8位图像）。

    返回:
        mse: 均方误差
        psnr: 峰值信噪比（dB）
        ssim_value: 结构相似性指数
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # 计算MSE
    mse = np.mean((img1 - img2) ** 2)

    # 计算PSNR
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * math.log10(max_pixel ** 2 / mse)

    # 计算SSIM
    if img1.ndim == 2:  # 灰度图像
        ssim_value = ssim(img1, img2, data_range=max_pixel)
    elif img1.ndim == 3:  # 彩色图像
        ssim_r = ssim(img1[:, :, 2], img2[:, :, 2], data_range=max_pixel)
        ssim_g = ssim(img1[:, :, 1], img2[:, :, 1], data_range=max_pixel)
        ssim_b = ssim(img1[:, :, 0], img2[:, :, 0], data_range=max_pixel)
        ssim_value = (ssim_r + ssim_g + ssim_b) / 3
    else:
        raise ValueError("输入图像维度不正确，应为2维或3维。")

    return mse, psnr, ssim_value


if __name__ == "__main__":
    # 使用tkinter打开文件对话框选择图像
    root = tkinter.Tk()
    root.withdraw()

    # 选择原始图像
    original_img_path = filedialog.askopenfilename(title="选择原始图像")
    original_img = cv2.imread(original_img_path)

    # 选择压缩图像
    compater_img_path = filedialog.askopenfilename(title="选择对比图像")
    compater_img = cv2.imread(compater_img_path)

    # 计算并打印结果
    mse, psnr, ssim_value = calculate_mse_psnr_ssim(original_img, compater_img)
    print(f"MSE: {mse}, PSNR: {psnr} dB, SSIM: {ssim_value}")

