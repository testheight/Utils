import torch,onnx
import sys,os
sys.path.append(r'D:\31890\Desktop\codefile\Utils')
from model__ import segformer_m    


net = segformer_m(2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.eval().to(device)
net.load_state_dict(torch.load(r'D:\31890\Desktop\codefile\Utils\Packages\onnx__\min_loss_model.pth',map_location=device)) 

## 生成onnx模型

x = torch.randn(1, 3, 512, 512)
torch_out = torch.onnx.export(net,
                            x,
                            "./test.onnx",
                            input_names=['input'],
                            output_names=['output'],
                            opset_version=11  #算子版本
                            )