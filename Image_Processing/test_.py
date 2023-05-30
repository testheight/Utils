import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
from numpy.random import randint as randint
 
def ellipse(img):
    h,w,c = img.shape

    center_x = randint(w//2-10, high=w//2+10)  #随机绘制圆心
    center_y = randint(h//2-10, high=h//2+10)
    X = randint(360, 380)//2
    Y = randint(460, 480)//2
    # 参数 1.目标图片  2.椭圆圆心  3.长短轴长度  4.偏转角度  5.圆弧起始角度  6.终止角度  7.颜色  8.是否填充
    cv2.ellipse(img, (center_x,center_y), (X,Y), randint(0,120), 0, 360, (255, 255, 255),-1)

def main():
    # 3.显示结果
    number =15
    max_shape=900
    min_shape=670
    for i in range(number):
        img = np.ones((randint(min_shape, max_shape),randint(min_shape, max_shape) , 3), np.uint8)*0
        ellipse(img)
 
        imgpath = './fig/' + str(i) + '.jpg'
        cv2.imwrite(imgpath, img)
    #     plt.clf()
    #     plt.close()
 
    # cv2.imshow("img", img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
 
 
if __name__ == '__main__':
    main()