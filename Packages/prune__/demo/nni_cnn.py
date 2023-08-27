# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import sys
sys.path.append(r'D:\31890\Desktop\codefile\Utils')
import torch
from model__ import segformer_m,segnet

from nni.compression.pytorch import TorchEvaluator
from nni.compression.pytorch.pruning import SlimPruner
# from nni.compression.pytorch import auto_set_denpendency_group_ids
from nni.compression.pytorch import ModelSpeedup
import segmentation_models_pytorch as smp

def unet_smp(num_classes=2):
    model = smp.Unet(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=num_classes,                      # model output channels (number of classes in your dataset)
    )
    return model

    



if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = unet_smp()

    model = segnet()
    # model = segformer_m()
    # model.load_state_dict(torch.load(r'D:\31890\Desktop\codefile\Utils\data\Image\min_loss_model.pth',map_location=device)) 
    
    config_list = [{
        'sparsity_per_layer': 0.7,
        'op_types': ['Linear', 'Conv2d']
    }, {
        'exclude': True,
        'op_partial_names': ['EfficientMultiHeadAttention']
    }]
    # config_list = [{
    # 'sparsity_per_layer': 0.5,
    # 'op_types': [ 'Conv2d']
    # }]
    # dummy_input = torch.rand(3, 3, 512, 512).to(device)
    dummy_input = torch.rand(1, 3, 224, 224).to(device)
    

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

    ModelSpeedup(model, dummy_input.to(device), masks).speedup_model()