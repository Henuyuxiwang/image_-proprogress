#基于平均亮度的自动gamma矫正
#作者：yuxi
#时间：2024/07/23
#邮箱：
#基于平均亮度的自动gamma矫正，论文《ASIC implementation of automatic gamma correction based on average of brightness 》
#假定一幅合理的图像应该所有像素的平均值应该是0.5左右（归一化后的），所以那么自动伽马校正的伽马值就要使得目标图像向这个目标
#防止偏色，需要平均各个通道亮度gamma


import cv2
import numpy as np
import math


def ASIC_IMAGE_GAMMA(IMAGE):
    # IMAGE = cv2.cvtColor(IMAGE, cv2.COLOR_BGR2GRAY)
    image=IMAGE

    if len(IMAGE.shape)  == 3:


        # image=image.astype(np.float32)
        normalized = image / 255
        # cv2.namedWindow('normalized', cv2.WINDOW_NORMAL)
        # cv2.imshow('normalized', normalized)
        mean_value = np.mean(normalized, axis=(0, 1))
        print(mean_value)
        Gamma_0 = -0.3 / (math.log10(mean_value[0]))
        Gamma_1 = -0.3 / (math.log10(mean_value[1]))
        Gamma_2 = -0.3 / (math.log10(mean_value[2]))
        print(Gamma_0)
        print(Gamma_1)
        print(Gamma_2)
        image[:, :, 0] = ((normalized[:, :, 0]) ** Gamma_0) * 255
        image[:, :, 1] = ((normalized[:, :, 1]) ** Gamma_1) * 255
        image[:, :, 2] = ((normalized[:, :, 2]) ** Gamma_2) * 255
        #原始论文中分通道偏色,解决方法有把三通道求得的Gamma值再求平均值，作为每个通道的Gamma值，也可以对亮度通道做Gamma，然后在返回到RGB空间等等。
        outIMAGE1 = image.astype(np.uint8)

        mean_gamma = (Gamma_0 + Gamma_1 + Gamma_2) / 3
        print(mean_gamma)
        image[:, :, 0] = ((normalized[:, :, 0]) ** mean_gamma) * 255
        image[:, :, 1] = ((normalized[:, :, 1]) ** mean_gamma) * 255
        image[:, :, 2] = ((normalized[:, :, 2]) ** mean_gamma) * 255

        outIMAGE2 = image.astype(np.uint8)

    else:
        normalized = image / 255
        # cv2.namedWindow('normalized', cv2.WINDOW_NORMAL)
        # cv2.imshow('normalized', normalized)
        mean_value = np.mean(normalized, axis=(0, 1))
        print(mean_value)
        Gamma_0 = -0.3 / (math.log10(mean_value))
        print(Gamma_0)
        image = (normalized ** Gamma_0) * 255
        outIMAGE1 = image.astype(np.uint8)
        outIMAGE2 = image.astype(np.uint8)


    return outIMAGE1,outIMAGE2


if __name__=="__main__":

    import tkinter as tk
    from tkinter import filedialog
    import time
    ##手动选择任意图像
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename()
    image = cv2.imread(path)
    #image=cv2.imread("zhongjian2562.jpg")

    time1=time.time()
    outIMAGE1,outIMAGE2=ASIC_IMAGE_GAMMA(image)
    time2=time.time()
    print("运行时间：",time2-time1)
    cv2.namedWindow('inIMAGE', cv2.WINDOW_NORMAL)
    cv2.imshow('inIMAGE', image)
    cv2.namedWindow('outIMAGE1', cv2.WINDOW_NORMAL)
    cv2.imshow('outIMAGE1', outIMAGE1)
    cv2.namedWindow('outIMAGE2', cv2.WINDOW_NORMAL)
    cv2.imshow('outIMAGE2', outIMAGE2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()