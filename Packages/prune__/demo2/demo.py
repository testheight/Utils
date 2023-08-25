import torch
import torch.nn.functional as F
from torch.optim import SGD

from nni_assets.compression.mnist_model import TorchModel, trainer, evaluator, device

# define the model
model = TorchModel().to(device)

# show the model structure, note that pruner will wrap the model layer.
print(model)

# define the optimizer and criterion for pre-training

optimizer = SGD(model.parameters(), 1e-2)
criterion = F.nll_loss

# pre-train and evaluate the model on MNIST dataset
for epoch in range(3):
    trainer(model, optimizer, criterion)
    evaluator(model)

torch.save(model.state_dict(),'Packages\prune__\demo\para1.pth')


config_list = [{
    'sparsity_per_layer': 0.5,
    'op_types': ['Linear', 'Conv2d']
}, {
    'exclude': True,
    'op_names': ['fc3']
}]

from nni.compression.pytorch.pruning import L1NormPruner
pruner = L1NormPruner(model, config_list)

# show the wrapped model structure, `PrunerModuleWrapper` have wrapped the layers that configured in the config_list.
print(model)

# compress the model and generate the masks
_, masks = pruner.compress()
# show the masks sparsity
for name, mask in masks.items():
    print(name, ' sparsity : ', '{:.2}'.format(mask['weight'].sum() / mask['weight'].numel()))


# need to unwrap the model, if the model is wrapped before speedup
pruner._unwrap_model()

# speedup the model, for more information about speedup, please refer :doc:`pruning_speedup`.
from nni.compression.pytorch.speedup import ModelSpeedup

ModelSpeedup(model, torch.rand(3, 1, 28, 28).to(device), masks).speedup_model()

print(model)
torch.save(model.state_dict(),'Packages\prune__\demo2\para2.pth')

optimizer = SGD(model.parameters(), 1e-2)
for epoch in range(3):
    trainer(model, optimizer, criterion)
torch.save(model.state_dict(),'Packages\prune__\demo2\para3.pth')