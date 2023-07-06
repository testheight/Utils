import tqdm,cv2,os
import numpy as np

def pixel_replacement(image_file,save_file):
    for name in tqdm.tqdm(os.listdir(image_file)):
        save_path = os.path.join(save_file,name)
        image_path = os.path.join(image_file,name)
        img = cv2.imread(image_path)
        img2 = np.where(img>0,1,0)
        cv2.imwrite(save_path,img2)

image_file = r'D:\31890\Desktop\data\anno\train_8'
save_file  = r'D:\31890\Desktop\data\anno\train_voc'
pixel_grayscale(image_file,save_file)