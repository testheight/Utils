import torch
import torch.nn.functional as F
from torch.optim import SGD

from nni_assets.compression.mnist_model import TorchModel, trainer, evaluator, device

# define the model
model = TorchModel().to(device)

# show the model structure, note that pruner will wrap the model layer.
print(model)

model.to(device=device)                                                       # ---将网络拷贝到deivce中--#
model.load_state_dict(torch.load('Packages\prune__\demo2\para3.pth',map_location=device))