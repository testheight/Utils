'''图像分割添加边框'''
import cv2
# ---------------------------------------------------#
#   裁剪：
#       img：图片（cv格式）
#       target_size: 目标尺寸(仅支持正方形)
#       file_name: 不含扩展名的文件名
#       pic_out_path: 输出文件夹
#       padding: 填充颜色(B,G,R)
# ---------------------------------------------------#
def clip(img, target_size, file_name, pic_out_path, padding=(0, 0, 0)):
    max_y, max_x = img.shape[0], img.shape[1]
    # 若不能等分，则填充至等分
    if max_x % target_size != 0:
        padding_x = target_size - (max_x % target_size)
        img = cv2.copyMakeBorder(img, 0, 0, 0, padding_x, cv2.BORDER_CONSTANT, value=padding)
        max_x = img.shape[1]
    if max_y % target_size != 0:
        padding_y = target_size - (max_y % target_size)
        img = cv2.copyMakeBorder(img, 0, padding_y, 0, 0, cv2.BORDER_CONSTANT, value=padding)
        max_y = img.shape[0]
 
    h_count = int(max_x / target_size)
    v_count = int(max_y / target_size)
 
    count = 0
    for v in range(v_count):
        for h in range(h_count):
            x_start = h * target_size
            x_end = (h + 1) * target_size
            y_start = v * target_size
            y_end = (v + 1) * target_size
            cropImg = img[y_start:y_end, x_start:x_end]  # 裁剪图像
            target_path = pic_out_path + file_name + '_' + str(count) + '.jpg'
            cv2.imwrite(target_path, cropImg)  # 写入图像路径
            count += 1


'''标注json转为图像'''
import argparse
import json
import os
import os.path as osp
import warnings
 
import PIL.Image
import yaml
 
from labelme import utils
import base64
 
def main():
    warnings.warn("This script is aimed to demonstrate how to convert the\n"
                  "JSON file to a single image dataset, and not to handle\n"
                  "multiple JSON files to generate a real-use dataset.")
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file',default='/mnt/d/31890/Desktop/codefile/Utils/data2')
    parser.add_argument('-o', '--out', default='/mnt/d/31890/Desktop/codefile/Utils/data_r')
    args = parser.parse_args()
 
    json_file = args.json_file
    if args.out is None:
        out_dir = osp.basename(json_file).replace('.', '_')
        out_dir = osp.join(osp.dirname(json_file), out_dir)
    else:
        out_dir = args.out
    if not osp.exists(out_dir):
        os.mkdir(out_dir)
 
    count = os.listdir(json_file) 
    for i in range(0, len(count)):
        path = os.path.join(json_file, count[i])
        if os.path.isfile(path):
            data = json.load(open(path))
            
            if data['imageData']:
                imageData = data['imageData']
            else:
                imagePath = os.path.join(os.path.dirname(path), data['imagePath'])
                with open(imagePath, 'rb') as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode('utf-8')
            img = utils.img_b64_to_arr(imageData)
            label_name_to_value = {'_background_': 0}
            for shape in data['shapes']:
                label_name = shape['label']
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value
            
            # label_values must be dense
            label_values, label_names = [], []
            for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
                label_values.append(lv)
                label_names.append(ln)
            assert label_values == list(range(len(label_values)))
            
            lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)
            
            captions = ['{}: {}'.format(lv, ln)
                for ln, lv in label_name_to_value.items()]
            lbl_viz = utils.draw_label(lbl, img, captions)
            
            out_dir = osp.basename(count[i]).replace('.', '_')
            out_dir = osp.join(osp.dirname(count[i]), out_dir)
            if not osp.exists(out_dir):
                os.mkdir(out_dir)
 
            PIL.Image.fromarray(img).save(osp.join(out_dir, 'img.png'))
            #PIL.Image.fromarray(lbl).save(osp.join(out_dir, 'label.png'))
            utils.lblsave(osp.join(out_dir, 'label.png'), lbl)
            PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, 'label_viz.png'))
 
            with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
                for lbl_name in label_names:
                    f.write(lbl_name + '\n')
 
            warnings.warn('info.yaml is being replaced by label_names.txt')
            info = dict(label_names=label_names)
            with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
                yaml.safe_dump(info, f, default_flow_style=False)
 
            print('Saved to: %s' % out_dir)

'''随机绘图'''
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
 
d=256
 
#随机绘制实心圆d
def cricle1(img):
    #d = 256
   # img = np.random.randint(0, 255) * np.ones((d, d, 3), np.uint8)
    for i in range(0, 2):
        # 随机中心点
        center_x = np.random.randint(0, high=d)
        center_y = np.random.randint(0, high=d)
        # 随机半径与颜色
        radius = np.random.randint(5, high= d /5)
        color = np.random.randint(0, high=256, size=(3, )).tolist()
        cv2.circle(img, (center_x, center_y), radius, color, -1)
     #   return 0
        # imgpath = './fig/' + str(i) + '1.jpg'
        # cv2.imwrite(imgpath, img)
 
#绘制空心圆
def cricle2(img):
    # d = 256
    # img = np.random.randint(0, 255) * np.ones((d, d, 3), np.uint8)
    for i in range(0, 2):
        # 随机中心点
        center_x = np.random.randint(0, high=d)
        center_y = np.random.randint(0, high=d)
        # 随机半径与颜色
        radius = np.random.randint(5, high= d /5)
        color = np.random.randint(0, high=256, size=(3, )).tolist()
        cv2.circle(img, (center_x, center_y), radius, color, 2)
      #  return 0
        # imgpath = './fig/' + str(i) + '2.jpg'
        # cv2.imwrite(imgpath, img)
 
 
#绘制矩形
def tangle(img):
    # d = 256
    # img = np.random.randint(0, 255) * np.ones((d, d, 3), np.uint8)
    n = np.random.randint(1,3)
    for i in range(n):
        long = np.random.randint(0, d)
        wide = np.random.randint(0, d)
        X = np.random.randint(0, d)
        Y = np.random.randint(0, d)
        color = np.random.randint(0, high=256, size=(3,)).tolist()
        cv2.rectangle(img,(X,Y),(long,wide), color,2)
      #  return 0
        # imgpath = './fig/' + str(i) + '3.jpg'
        # cv2.imwrite(imgpath, img)
 
 
#绘制椭圆
def ellipse(img):
    # d = 256
    # img = np.random.randint(0, 255) * np.ones((d, d, 3), np.uint8)
    n = np.random.randint(1, 3)
    for i in range(n):
        center_x = np.random.randint(0, high=d)  #随机绘制圆心
        center_y = np.random.randint(0, high=d)
        X = np.random.randint(0, d/5)
        Y = np.random.randint(0, d/5)
        color = np.random.randint(0, high=256, size=(3,)).tolist()
        cv2.ellipse(img, (center_x,center_y), (X,Y), 0, 0, 360, color,2)
       # return 0
        # imgpath = './fig/' + str(i) + '4.jpg'
        # cv2.imwrite(imgpath, img)
 
 
def main():
    # 3.显示结果
    figurenum =50
    # d=256
    # img = np.random.randint(0, 255) * np.ones((d, d, 3), np.uint8)
    for i in range(figurenum):
        img = np.random.randint(0, 255) * np.ones((d, d, 3), np.uint8)
        a = np.random.randint(0, 3)
        if a == 0:
            cricle1(img)
        elif a == 1:
            cricle2(img)
        elif a == 2:
            tangle(img)
        elif a == 3:
            ellipse(img)
 
        imgpath = './fig/' + str(i) + '.jpg'
        cv2.imwrite(imgpath, img)
        plt.clf()
        plt.close()
 
    cv2.imshow("img", img)
    
    cv2.waitKey()
    cv2.destroyAllWindows()
