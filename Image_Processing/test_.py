import cv2,os,sklearn,tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans,AgglomerativeClustering,MeanShift,SpectralClustering
from utils import color
from PIL import Image


def Clustering(img_array,method="kmeans"):
    h,w,c = img_array.shape

    img_array =  img_array.reshape(-1,3).astype(np.uint8)
    if method == "kmeans":
        classifier = KMeans(n_clusters=2,n_init="auto",random_state=0,init='k-means++')
    elif method == "AgglomerativeClustering":               # 不适用,计算量太大
        classifier = AgglomerativeClustering(n_clusters=2)
    elif method == "MeanShift":                             # 耗时过长
        classifier = MeanShift(bandwidth=2)
    elif method == "SpectralClustering":                    # 不适用,计算量太大
        classifier = SpectralClustering(n_clusters=2)
        
    kmeans = classifier.fit(img_array)
    img_label = kmeans.labels_
    img_label = img_label.reshape(h,w)

    return img_label




img_file = r"D:\31890\Desktop\codefile\Utils\data"
save_file = r'D:\31890\Desktop\codefile\Utils\result'

if not os.path.exists(save_file):
    os.makedirs(save_file)

for name in tqdm.tqdm(os.listdir(img_file)):
    image_path = os.path.join(img_file,name)
    save_path = os.path.join(save_file,name)

    image_array = cv2.imread(image_path)
    img_label = Clustering(image_array,method="kmeans")
    img_label = color(img_label,colormap=[0, 0, 0,255,255,255])
    img_label.save(save_path)


