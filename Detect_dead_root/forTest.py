import copy

import matplotlib.pyplot as plt  # plt 用于显示图片
import numpy as np
import cv2


def filterMinMax(imgRaw, weight):
    dst = copy.deepcopy(imgRaw)

    info1 = imgRaw.shape
    height1 = info1[0]
    width1 = info1[1]
    lenx = len(imgRaw[1, 1])

    for h in range(0, height1):
        for j in range(0, width1):

            if lenx == 4:
                b, g, r, _ = imgRaw[h, j]
            else:
                b, g, r = imgRaw[h, j]

            if (b > range1Min[0]) & (b < range1Max[0]) & (g > range1Min[1]) & (g < range1Max[1]) & (
                    r > range1Min[2]) & (r < range1Max[
                2]):

                if lenx == 4:
                    dst[h, j] = [0, 0, 0, 255]
                else:
                    dst[h, j] = [0, 0, 0]

            if (b > range2Min[0]) & (b < range2Max[0]) & (g > range2Min[1]) & (g < range2Max[1]) & (
                    r > range2Min[2]) & (r < range2Max[
                2]):
                dst[h, j] = [0, 0, 0, 255]

            # if (b < range2Max[0]) & (g < range2Max[1]) & (r < range2Max[2]):
            #     dst[h, j] = [0, 0, 0, 255]

            if (b < range2Max[0]) | (g < range2Max[1]) | (r < range2Max[2]):
                dst[h, j] = [0, 0, 0, 255]

            if (b < range1Max[0]) | (g < range1Max[1]) | (r < range1Max[2]):
                dst[h, j] = [0, 0, 0, 255]

            if (b < range0Min[0]) | (g < range0Min[1]) | (r < range0Min[2]):  # 这个是最管用的  去掉death
                dst[h, j] = [0, 0, 0, 255]


img = cv2.imread("findPixDis/1.png", cv2.IMREAD_UNCHANGED)  #
img1 = cv2.imread("findPixDis/2.png", cv2.IMREAD_UNCHANGED)  #
img2 = cv2.imread("findPixDis/3.png", cv2.IMREAD_UNCHANGED)  #

info = img.shape
height = info[0]
width = info[1]

info1 = img1.shape
height1 = info1[0]
width1 = info1[1]

# 选择一个区域,判断他们的均值
x1 = []
x2 = []
x3 = []
for h in range(10, 30):
    for j in range(10, 30):
        x1.append(img[h, j])
        x2.append(img1[h, j])
        x3.append(img2[h, j])

x1 = np.array(x1)
x2 = np.array(x2)
x3 = np.array(x3)

mean1 = np.mean(x1, axis=0)
mean2 = np.mean(x2, axis=0)
mean3 = np.mean(x3, axis=0)

std1 = np.std(x1, axis=0)
std2 = np.std(x2, axis=0)
std3 = np.std(x3, axis=0)

# 求范围
range0Min, range0Max = mean1 - std1, mean1 + std1  # 活的范围
range1Min, range1Max = mean2 - std2, mean2 + std2  # 死的范围
range2Min, range2Max = mean3 - std3, mean3 + std3  # 死的范围

imgRaw = cv2.imread("test/RawImagePart.png", cv2.IMREAD_UNCHANGED)  #

dst = copy.deepcopy(imgRaw)

info1 = imgRaw.shape
height1 = info1[0]
width1 = info1[1]

"""对死根进行过滤"""
for h in range(0, height1):
    for j in range(0, width1):
        b, g, r,_  = imgRaw[h, j]
        if (b > range1Min[0]) & (b < range1Max[0]) & (g > range1Min[1]) & (g < range1Max[1]) & (r > range1Min[2]) & (
                r < range1Max[
            2]):
            dst[h, j] = [0, 0, 0, 255]

        if (b > range2Min[0]) & (b < range2Max[0]) & (g > range2Min[1]) & (g < range2Max[1]) & (r > range2Min[2]) & (
                r < range2Max[
            2]):
            dst[h, j] = [0, 0, 0, 255]

        # if (b < range2Max[0]) & (g < range2Max[1]) & (r < range2Max[2]):
        #     dst[h, j] = [0, 0, 0, 255]

        if (b < range2Max[0]) | (g < range2Max[1]) | (r < range2Max[2]):
            dst[h, j] = [0, 0, 0, 255]

        if (b < range1Max[0]) | (g < range1Max[1]) | (r < range1Max[2]):
            dst[h, j] = [0, 0, 0, 255]

        if (b < range0Min[0]) | (g < range0Min[1]) | (r < range0Min[2]):  # 这个是最管用的  去掉death
            dst[h, j] = [0, 0, 0, 255]

cv2.imwrite("test/RawImagePartb1.png", dst)  #

# 对dst进行均值滤波
size = 5
k = np.ones([size, size])
k = k / (size * size)  # 均值滤波器

img_high2 = cv2.filter2D(dst, -1, k)  #
cv2.imwrite("test/RawImagePartb2.png", img_high2)

# 锐化
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
dst2 = cv2.filter2D(img_high2, -1, kernel=kernel)
cv2.imwrite("test/RawImagePartb3.png", dst2)

# 在进行一次滤波
for h in range(0, height1):
    for j in range(0, width1):
        b, g, r, _ = dst2[h, j]
        if (b < range0Min[0] / 3) | (g < range0Min[1] / 3) | (r < range0Min[2] / 3):  # 这个是最管用的  去掉death
            dst2[h, j] = [0, 0, 0, 255]

cv2.imwrite("test/RawImagePartb4.png", dst2)

"""和原始图像比较 并进行增强"""
ki = 2  # 定义核的大小为3*3
img = cv2.imread("test/RawImagePartb4.png", cv2.IMREAD_UNCHANGED)  # 已经去掉一部分的图像
img1 = cv2.imread("test/RawImagePart.png", cv2.IMREAD_UNCHANGED)  # 原始的图像

dstb5 = copy.deepcopy(img)
img2 = copy.deepcopy(img1)

img[img == 255] = 0
img2[img2 > 0] = 0  # 和全黑的比

height = img.shape[0]
width = img.shape[1]

for i in range(height):
    for j in range(width):
        h1 = i - ki if (i - ki) > 0 else 0
        h2 = i + ki if (i + ki) < height else height
        w1 = j - ki if (j - ki) > 0 else 0
        w2 = j + ki if (i + ki) < width else width

        x = np.array(img[h1:h2, w1:w2]).reshape(1, -1)
        x1 = np.array(img2[h1:h2, w1:w2]).reshape(1, -1)

        # print(np.array(x).reshape(1,-1)[0],np.array(x1).reshape(1,-1)[0])
        dis = np.linalg.norm(x - x1)
        # if dis > 0: print(dis)

        # if dis < 400:
        #     dstb5[h1:h2, w1:w2] = imgRaw[h1:h2, w1:w2]

        if dis > 0:
            dstb5[h1:h2, w1:w2] = imgRaw[h1:h2, w1:w2]

cv2.imwrite("test/RawImagePartb5.png", dstb5)  #
