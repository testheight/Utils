import shutil
import os,tqdm,random
 
def remove_file(image_file, img_save_file):
    filelist = os.listdir(image_file) 
    for file in tqdm.tqdm(filelist):
        for name in os.listdir(os.path.join(image_file,file)):
            old_path = os.path.join(image_file,file,name)
            new_path = os.path.join(img_save_file,name)
            shutil.move(old_path, new_path)

def remove_file2(image_file, mask_file, img_save_file,mask_save_file):
    filelist = os.listdir(image_file) 
    val_List =random.sample(filelist,k = int(len(filelist)*0.2))
    for name in tqdm.tqdm(val_List):
        img_old_path = os.path.join(image_file,name)
        img_new_path = os.path.join(img_save_file,name)
        mask_old_path = os.path.join(mask_file,name.split('.jp')[0]+'.png')
        mask_new_path = os.path.join(mask_save_file,name.split('.jp')[0]+'.png')
        shutil.move(img_old_path, img_new_path)
        shutil.move(mask_old_path, mask_new_path)

if __name__ == "__main__":
    # image_file = r'D:\31890\Desktop\PRMI_official\test\masks_pixel_gt'
    # img_save_file = r'D:\31890\Desktop\data\anno\test'
    # remove_file(image_file, img_save_file)

    image_file = r'D:\31890\Desktop\mseg_root_data\imgs\test'
    mask_file =  r'D:\31890\Desktop\mseg_root_data\anno\test'
    img_save_file = r'D:\31890\Desktop\mseg_mix_data\imgs\test'
    mask_save_file = r'D:\31890\Desktop\mseg_mix_data\anno\test'
    remove_file2(image_file,mask_file, img_save_file,mask_save_file)