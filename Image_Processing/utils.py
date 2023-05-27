import os,tqdm,cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sklearn.cluster 
import pandas as pd


#索引图上色
def color(image_array,colormap=[0, 0, 0,0, 128, 0, 128, 128, 0]):
    '''
    image_array: PIL格式 灰度图或者
    '''
    if isinstance(image_array, np.ndarray):
        image_array = Image.fromarray(image_array)
        
    image_array = image_array.convert('L')
    #调色板
    palette = colormap
    #着色
    image_array.putpalette(palette)
    
    return image_array
