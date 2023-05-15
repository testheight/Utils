# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.datasets import load_iris
 
# data = load_iris()
# x = data.data
# y = data.target
# pca = PCA(n_components=2)
# reduced_X = pca.fit_transform(x)
 
# red_x, red_y = [], []
# blue_x, blue_y = [], []
# green_x, green_y = [], []
# for i in range(len(reduced_X)):
#     if y[i] == 0:
#         red_x.append(reduced_X[i][0])
#         red_y.append(reduced_X[i][1])
#     elif y[i] == 1:
#         blue_x.append(reduced_X[i][0])
#         blue_y.append(reduced_X[i][1])
#     else:
#         green_x.append(reduced_X[i][0])
#         green_y.append(reduced_X[i][1])
# plt.scatter(red_x, red_y, c='r', marker='x')
# plt.scatter(blue_x, blue_y, c='b', marker='D')
# plt.scatter(green_x, green_y, c='g', marker='.')
# plt.show()

import numpy
import cv2
import tqdm
import argparse
 
# 将原作者的sys转换成params
parser = argparse.ArgumentParser(description='SLIC-python')
parser.add_argument('--img_path', default=r'D:\31890\Desktop\codefile\Utils\Image\2-2019-7-08-1200_15_10.jpg', type=str, help="单张图片的路径")
parser.add_argument('--k', default=120, type=int, help="超像素个数")
parser.add_argument('--SLIC_ITERATIONS', default=4, type=int, help="SLIC计算过程中的迭代次数")
parser.add_argument('--m', default=40, type=int, help="权衡颜色和位置对距离影响的权重参数")
args = parser.parse_args()
 
 
def generate_pixels():
    indnp = numpy.mgrid[0:SLIC_height, 0:SLIC_width].swapaxes(0, 2).swapaxes(0, 1)
    # 迭代SLIC_ITERATIONS次
    for i in tqdm.tqdm(range(SLIC_ITERATIONS)):
        SLIC_distances = 1 * numpy.ones(img.shape[:2])
        # 按次序取出聚类中心SLIC_centers[j]
        for j in range(SLIC_centers.shape[0]):
            # 框出该聚类中心的搜索范围
            x_low, x_high = int(SLIC_centers[j][3] - step), int(SLIC_centers[j][3] + step)
            y_low, y_high = int(SLIC_centers[j][4] - step), int(SLIC_centers[j][4] + step)
 
            # 防止搜索范围超出图像边界[保证搜索范围有效性]
            if x_low <= 0:
                x_low = 0
            if x_high > SLIC_width:
                x_high = SLIC_width
            if y_low <= 0:
                y_low = 0
            if y_high > SLIC_height:
                y_high = SLIC_height
 
            # cropimg是该聚类中心对应的2S\times2S内的有效邻域
            cropimg = SLIC_labimg[y_low: y_high, x_low: x_high]
            # 挨个像素算出颜色差
            color_diff = cropimg - SLIC_labimg[int(SLIC_centers[j][4]), int(SLIC_centers[j][3])]
            # 算出颜色距离
            color_distance = numpy.sqrt(numpy.sum(numpy.square(color_diff), axis=2))
 
            yy, xx = numpy.ogrid[y_low: y_high, x_low: x_high]
            # 算出空间距离
            pixdist = ((yy - SLIC_centers[j][4]) ** 2 + (xx - SLIC_centers[j][3]) ** 2) ** 0.5
 
            # 运用论文中的(2)式计算邻域内pixel与该邻域中心的聚类中心的距离（加权求和）
            # SLIC_m is "m" in the paper, (m/S)*dxy
            dist = ((color_distance / SLIC_m) ** 2 + (pixdist / step) ** 2) ** 0.5
 
            # 更新距离，更新了距离的pixel也更新聚类中心为SLIC_centers[j]
            distance_crop = SLIC_distances[y_low: y_high, x_low: x_high]
            idx = dist < distance_crop
            distance_crop[idx] = dist[idx]
            SLIC_distances[y_low: y_high, x_low: x_high] = distance_crop
            SLIC_clusters[y_low: y_high, x_low: x_high][idx] = j
 
        for k in range(len(SLIC_centers)):
            # 对于第k个聚类，找到聚类中心为SLIC_centers[k]的pixel
            idx = (SLIC_clusters == k)
            # 分别取出他们的颜色和位置索引
            colornp = SLIC_labimg[idx]
            distnp = indnp[idx]
 
            # 重新计算聚类中心的颜色和位置坐标（这个聚类中心和k-means中的一样，不一定是已有的点）
            SLIC_centers[k][0:3] = numpy.sum(colornp, axis=0)
            sumy, sumx = numpy.sum(distnp, axis=0)
            SLIC_centers[k][3:] = sumx, sumy
            ### 注：numpy.sum(idx)是该聚类pixel数目
            SLIC_centers[k] /= numpy.sum(idx)
 
 
# At the end of the process, some stray labels may remain meaning some pixels
# may end up having the same label as a larger pixel but not be connected to it
# In the SLIC paper, it notes that these cases are rare, however this
# implementation seems to have a lot of strays depending on the inputs given
 
def create_connectivity():
    """
        按照论文的说法，总有那么些点和它对应的超像素是分离的（比较零散的碎点）
        运用connected components algorithm来将这些零散的点分配给最近的聚类中心
    """
    label = 0
    adj_label = 0
    lims = int(SLIC_width * SLIC_height / SLIC_centers.shape[0])
 
    new_clusters = -1 * numpy.ones(img.shape[:2]).astype(numpy.int64)
    elements = []
    for i in range(SLIC_width):
        for j in range(SLIC_height):
            if new_clusters[j, i] == -1:
                elements = []
                elements.append((j, i))
                for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                    x = elements[0][1] + dx
                    y = elements[0][0] + dy
                    if (x >= 0 and x < SLIC_width and
                            y >= 0 and y < SLIC_height and
                            new_clusters[y, x] >= 0):
                        adj_label = new_clusters[y, x]
                    # end
                # end
            # end
 
            count = 1
            counter = 0
            while counter < count:
                for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                    x = elements[counter][1] + dx
                    y = elements[counter][0] + dy
 
                    if (x >= 0 and x < SLIC_width and y >= 0 and y < SLIC_height):
                        if new_clusters[y, x] == -1 and SLIC_clusters[j, i] == SLIC_clusters[y, x]:
                            elements.append((y, x))
                            new_clusters[y, x] = label
                            count += 1
                        # end
                    # end
                # end
 
                counter += 1
            # end
            if (count <= lims >> 2):
                for counter in range(count):
                    new_clusters[elements[counter]] = adj_label
                # end
 
                label -= 1
            # end
 
            label += 1
        # end
    # end
 
    SLIC_new_clusters = new_clusters
 
 
# end
 
def display_contours(color):
    is_taken = numpy.zeros(img.shape[:2], numpy.bool)  # 标志哪些点是聚类与聚类之间的edge
    contours = []
 
    for i in range(SLIC_width):
        for j in range(SLIC_height):
            nr_p = 0
            for dx, dy in [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]:
                x = i + dx
                y = j + dy
                if x >= 0 and x < SLIC_width and y >= 0 and y < SLIC_height:
                    if is_taken[y, x] == False and SLIC_clusters[j, i] != SLIC_clusters[y, x]:
                        nr_p += 1
                    # end
                # end
            # end
 
            if nr_p >= 2:
                is_taken[j, i] = True
                contours.append([j, i])
 
    # 将这些edge-pixel全用黑色来表示
    for i in range(len(contours)):
        img[contours[i][0], contours[i][1]] = color
        mask[contours[i][0], contours[i][1]] = color
    # end
 
 
# end
 
def find_local_minimum(center):
    """
        微调
        在3\times3领域内找梯度最小的点作为初始聚类中心
    """
    min_grad = 1
    loc_min = center
    for i in range(center[0] - 1, center[0] + 2):
        for j in range(center[1] - 1, center[1] + 2):
            c1 = SLIC_labimg[j + 1, i]
            c2 = SLIC_labimg[j, i + 1]
            c3 = SLIC_labimg[j, i]
            if ((c1[0] - c3[0]) ** 2) ** 0.5 + ((c2[0] - c3[0]) ** 2) ** 0.5 < min_grad:
                min_grad = abs(c1[0] - c3[0]) + abs(c2[0] - c3[0])
                loc_min = [i, j]
    return loc_min
 
 
def calculate_centers():
    """
        按照grid_cell初始化聚类中心
    """
    centers = []
    for i in range(step, SLIC_width - int(step / 2), step):
        for j in range(step, SLIC_height - int(step / 2), step):
            nc = find_local_minimum(center=(i, j))  # 微调
            color = SLIC_labimg[nc[1], nc[0]]
            center = [color[0], color[1], color[2], nc[0], nc[1]]  # LAB+XY
            centers.append(center)
    return centers  # 储存聚类中心的信息
 
 
# 样例命令是slic.py Lenna.png 1000 40
# sys.argv[1]是放图片路径
# sys.argv[2]这个参数指示划分的superpixel的个数
# sys.argv[3]这个参数是论文中的m与论文中的m对应，是计算点与点间的距离时用于衡量颜色距离和空间距离所占权重的重要参数
 
# global variables
img = cv2.imread(args.img_path)
mask = 255 * numpy.ones(img.shape).astype('uint8')
step = int((img.shape[0] * img.shape[1] / args.k) ** 0.5)  # 每个superpixel中心之间的平均距离
SLIC_m = args.m
SLIC_ITERATIONS = args.SLIC_ITERATIONS  # 迭代次数
SLIC_height, SLIC_width = img.shape[:2]
SLIC_labimg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(numpy.float64)  # BGR转LAB
 
# 初始化距离和每个点所属聚类中心
SLIC_distances = 1 * numpy.ones(img.shape[:2])
SLIC_clusters = -1 * SLIC_distances  ### 我们应该是依靠这个搞出mask ###
 
# 聚类中心初始化
SLIC_center_counts = numpy.zeros(len(calculate_centers()))
SLIC_centers = numpy.array(calculate_centers())
 
# main
generate_pixels()  # 迭代SLIC_ITERATIONS次，聚好各组点，算出他们的聚类中心位置和类颜色
create_connectivity()  # 后处理，对一些比较零散的点重新分配给邻近的聚类
calculate_centers()
display_contours([0.0, 0.0, 0.0])
img2 = numpy.hstack((img, mask))
cv2.imwrite(r'D:\31890\Desktop\codefile\Utils\Image\3.jpg',img2)
 