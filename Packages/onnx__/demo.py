import torch,onnx
import sys,os
sys.path.append(r'D:\31890\Desktop\codefile\Utils')
from model__ import U_Net_o

## 生成onnx模型
net = U_Net_o(2)
x = torch.randn(1, 3, 512, 512)
torch_out = torch.onnx.export(net,
                            x,
                            "test.onnx",
                            verbose=True,
                            export_params=True
                            )