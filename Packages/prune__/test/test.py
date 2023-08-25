# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute().parents[1]))

import torch
from model__ import segformer_m  

from nni.compression.pytorch import TorchEvaluator
from nni.compression.pytorch.pruning import SlimPruner
# from nni.compression.pytorch import auto_set_denpendency_group_ids
from nni.compression.pytorch import ModelSpeedup


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = segformer_m()


    config_list = [{
        'op_types': ['Conv2d','Linear'],
        'sparse_ratio': 0.7
    }]
    dummy_input = torch.rand(3, 3, 512, 512).to(device)