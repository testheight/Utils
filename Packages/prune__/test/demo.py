# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
from unet_T import U_Net_o,U_Net_Multi2_add

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = U_Net_Multi2_add(2)

model.load_state_dict(torch.load('unet_Multi2_add.pth',map_location=device))  #unet_o

config_list = [{
'sparsity_per_layer': 0.2,
'op_types': [ 'Conv2d']
}]

dummy_input = torch.rand(3, 3, 512, 512)

print(model)
y = model(dummy_input)
print(y.shape)


from nni.compression.pytorch.pruning import L1NormPruner
pruner = L1NormPruner(model, config_list)


# compress the model and generate the masks
_, masks = pruner.compress()
# show the masks sparsity
for name, mask in masks.items():
    print(name, ' sparsity : ', '{:.2}'.format(mask['weight'].sum() / mask['weight'].numel()))


# need to unwarp the model, if the model is wrawpped before speedup
pruner._unwrap_model()

# speedup the model
model.eval()

from nni.compression.pytorch.speedup import ModelSpeedup

ModelSpeedup(model, dummy_input, masks).speedup_model()

torch.save(model,'unet_Multi2_add_prune.pt')