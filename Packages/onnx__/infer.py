import onnxruntime
import numpy as np
import torch,cv2,time,tqdm
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# 加载
t_session = onnxruntime.InferenceSession(r'D:\31890\Desktop\codefile\Utils\Packages\onnx__\test.onnx')

time_start = time.time()
img_path = r'D:\31890\Desktop\codefile\Utils\Packages\onnx__\test.jpg'
img_pil = Image.open(img_path)

# 输出图像的尺寸
pixel_shape = 512
origin_w, origin_h = img_pil.size
# 新的高和宽
new_h = int((origin_h//pixel_shape+1)*pixel_shape)
new_w = int((origin_w//pixel_shape+1)*pixel_shape)
# 多余的尺寸
h_padding = new_h-origin_h
w_padding = new_w-origin_w

# 创建
img_new = Image.new('RGB',(new_w,new_h))
img_new.paste(img_pil, (w_padding, h_padding))  ##上左田间黑边
mask = np.zeros((new_h,new_w))

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.4754358, 0.35509014, 0.282971],
            std=[0.16318515, 0.15616792, 0.15164918])
            ])

img_new = transform(img_new).unsqueeze(0).numpy()

for u in tqdm.tqdm(range(origin_h//pixel_shape)):
    for v in range(origin_w//pixel_shape):
        x = pixel_shape * u
        y = pixel_shape * v

        input_tensor = img_new[:,:,x : x + pixel_shape, y : y + pixel_shape]
        # ONNX Runtime 输入
        ort_inputs = {'input': input_tensor}
        # ONNX Runtime 输出
        pred_logits = t_session.run(['output'], ort_inputs)[0]
        pred_logits = torch.tensor(pred_logits)
        pred_logits = F.softmax(pred_logits[0],dim=0)
        pred = np.array(pred_logits.argmax(axis=0),)*255
        mask[x : x + pixel_shape, y : y + pixel_shape] = pred

mask = Image.fromarray(mask[:origin_h,:origin_w].astype(np.uint8))
mask.save(r'D:\31890\Desktop\codefile\Utils\Packages\onnx__\test2.png')

time_end = time.time()
print("time cost:", time_end-time_start , "s")