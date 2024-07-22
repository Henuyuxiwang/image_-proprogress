#全局直方图二值化，孔洞填充
#作者：yuxi
#时间：2024/07/22
#邮箱：
# 填充前景孔洞，和填充背景孔洞

import numpy as np
import cv2
import math


def otsu_threshold(image):
    if len(image.shape) >= 3:
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    binary_ret, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.namedWindow("Binary Image (OSTU)", cv2.WINDOW_NORMAL)
    # cv2.imshow("Binary Image (OSTU)", binary_image)
    # cv2.waitKey(100)
    # cv2.destroyAllWindows()

    return binary_ret, binary_image


def FillHole(src, fill_background):

    # 获取图像的宽和高

    width, height = src.shape

    pixels = src

    color = 255 if not fill_background else 0


    for y in range(height):
        if pixels[0][y] == color:
            pixels=flood_fill(pixels, (0, y), 127)
        if pixels[width - 1][y] == color:
            pixels=flood_fill(pixels,(width - 1, y), 127)

    for x in range(width):
        if pixels[x][0] == color:
            pixels=flood_fill(pixels,(x, 0), 127)
        if pixels[x][height - 1] == color:
            pixels=flood_fill(pixels, (x, height - 1), 127)

    for y in range(height):
        for x in range(width):
            if pixels[x, y] == 127:
                pixels[x, y] = color
            else:
                pixels[x, y] = 255 - color
    return pixels





def get_hist_gram(src):
    hist_gram = np.zeros(256, dtype=np.int32)
    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            hist_gram[src[y, x]] += 1
    return hist_gram


def get_huang_fuzzy_threshold(hist_gram):
    first, last = 0, len(hist_gram) - 1
    while first < len(hist_gram) and hist_gram[first] == 0:
        first += 1
    while last > first and hist_gram[last] == 0:
        last -= 1
    if first == last:  # 图像中只有一个颜色
        return first
    if first + 1 == last:  # 图像中只有二个颜色
        return first

    # 计算累计直方图以及对应的带权重的累计直方图
    s = [0] * (last + 1)
    w = [0] * (last + 1)  # 对于特大图，此数组的保存数据可能会超出int的表示范围，可以考虑用long类型来代替
    s[0] = hist_gram[0]
    for y in range(first > 1 and first or 1, last + 1):
        s[y] = s[y - 1] + hist_gram[y]
        w[y] = w[y - 1] + y * hist_gram[y]

    # 建立公式（4）及（6）所用的查找表
    smu = [0] * (last + 1 - first)
    for y in range(1, len(smu)):
        mu = 1 / (1 + y / (last - first))  # 公式（4）
        smu[y] = -mu * math.log(mu) - (1 - mu) * math.log(1 - mu)  # 公式（6）

    # 迭代计算最佳阈值
    BestEntropy = float('inf')
    Threshold = None
    for Y in range(first, last):
        Entropy = 0
        mu = round(w[Y] / s[Y])  # 公式17
        for X in range(first, Y + 1):
            Entropy += smu[abs(X - mu)] * HistGram[X]
        # mu = round((w[last] - w[Y]) / (s[last] - s[Y]))  # 公式18
        if not math.isnan(s[last] - s[Y]):
            mu = round((w[last] - w[Y]) / (s[last] - s[Y]))  # 公式18
        else:
            # 处理方式，例如跳过这次计算或者设置 mu 为一个默认值
            mu = 0  # 或者其他合适的默认值

        for X in range(Y + 1, last):
            Entropy += smu[abs(X - mu)] * HistGram[X]  # 公式8
        if BestEntropy > Entropy:
            BestEntropy = Entropy  # 取最小熵处为最佳阈值
            Threshold = Y
    return Threshold


def flood_fill(img, start_point, fill_value):
    # 获取图像的高度和宽度
    height, width = img.shape[:2]

    # 获取起始位置的颜色
    x, y = start_point
    target_color = img[x, y].tolist()  # 获取起始像素的颜色

    # 如果起始位置的颜色与填充颜色相同，则返回
    if target_color == fill_value:
        return img

    # 创建一个掩码，用于记录已填充的区域
    mask = np.zeros((height + 2, width + 2), np.uint8)

    # 创建栈以存储待处理的像素位置
    fill_points = [(x, y)]

    while fill_points:
        # 从栈中弹出一个像素位置
        px, py = fill_points.pop()

        # 填充当前像素
        img[px, py] = fill_value

        # 检查四个方向的相邻像素
        # 左侧
        if py > 0 and (img[px, py - 1].tolist() == target_color):
            fill_points.append((px , py- 1))
        # 右侧
        if py < width - 1 and (img[px, py + 1].tolist() == target_color):
            fill_points.append((px , py+ 1))
        # 上方
        if px > 0 and (img[px - 1, py].tolist() == target_color):
            fill_points.append((px- 1, py ))
        # 下方
        if px < height - 1 and (img[px + 1, py].tolist() == target_color):
            fill_points.append((px + 1, py))

    return img


if __name__ == "__main__":
    # img_path = r"D:\\SEM_001\\RulerTools1\\dispose2024\\dispose2024_v7.0\\dispose2024\\zx-0.tif"
    img_path=r"D:\\SEM_001\\RulerTools1\\dispose2024\\dispose2024_v7.0\\dispose2024\\img001.x-png"
    # Read image
    proimg = cv2.imread(img_path)
    if len(proimg.shape) >= 3:
        # Convert image to grayscale
        progray = cv2.cvtColor(proimg, cv2.COLOR_BGR2GRAY)
    else:
        progray = proimg

    HistGram=get_hist_gram(progray)
    threshold_value=get_huang_fuzzy_threshold(HistGram)
    print(threshold_value)
    ret_out, thresh_out = cv2.threshold(progray, threshold_value, 255, cv2.THRESH_BINARY)
    cv2.imshow("Foreground", thresh_out)
    cv2.waitKey(0)
    FillBackGround = 1

    img_out=FillHole(thresh_out,FillBackGround)

    # cv2.imshow("Foreground1", asas)
    # cv2.imwrite(r"D:\\SEM_001\\RulerTools1\\dispose2024\\dispose2024_v7.0\\dispose2024\\zx-0-out.tif", img_out)
    cv2.imshow("Foreground1", img_out)
    cv2.waitKey(0)


    # kernel = np.ones((3, 3))
    # proimg = cv2.dilate(progray, kernel)
    # th, im_th = otsu_threshold(proimg)
    #
    #
    # cv2.imshow("Foreground", im_th)
    # cv2.waitKey(0)
