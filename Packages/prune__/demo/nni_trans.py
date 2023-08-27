# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import sys
sys.path.append(r'D:\31890\Desktop\codefile\Utils')
import torch
from model__ import segnet,segformer_m
from typing import Callable, Any
import torch,os,cv2
import albumentations as A
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader,Dataset


import nni
from nni.algorithms.compression.v2.pytorch import TorchEvaluator
from nni.compression.pytorch import ModelSpeedup
from nni.compression.pytorch.pruning import MovementPruner
from torch.optim import Adam,AdamW
from torch.optim.lr_scheduler import StepLR


class train_Dataset(Dataset):
    def __init__(self,data_dir,input_size):
        #建立数据列表
        images,labels = [],[]
        for name in os.listdir(os.path.join(data_dir,'imgs','train')):
                images.append(os.path.join(data_dir,'imgs','train',name))
                labels.append(os.path.join(data_dir,'anno','train',name.split('.jp')[0]+'.png'))
        self.labels = labels 
        self.images = images
        self.input_size = input_size #list[]

    #获取图像    
    def __getitem__(self, index):
        img_path,label_path =  self.images[index],self.labels[index]
        #读取图像
        imgs = cv2.imread(img_path)
        imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
        lbls = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)
        #数据增强
        transform = A.Compose([
                A.RandomResizedCrop(height=self.input_size[0],width=self.input_size[1],scale=(0.15, 1.0)),    #旋转
                A.Rotate(p=0.3),                                                #翻转
                A.HorizontalFlip(p=0.3),                                        #水平翻转
                A.VerticalFlip(p=0.2),                                          #垂直翻转
                A.OneOf([
                    #随机改变图像的亮度、对比度和饱和度
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.1),
                    #随机改变输入图像的色调、饱和度和值
                    A.HueSaturationValue(p=0.3),
                    ],p=0.2),
                A.Normalize (mean=[0.4754358, 0.35509014, 0.282971],std=[0.16318515, 0.15616792, 0.15164918]),
                ToTensorV2(),
                ])
        transformed = transform(image=imgs, mask=lbls)
        imgs = transformed['image']
        lbls = transformed['mask'].long()
        return imgs,lbls
    
    #获取数据集长度
    def __len__(self):
        return len(self.images)

class test_Dataset(Dataset):
    def __init__(self,data_dir):
        self.data_dir = data_dir
        #建立数据列表
        images,labels = [],[]
        for name in os.listdir(os.path.join(data_dir,'imgs','test')):
                images.append(os.path.join(data_dir,'imgs','test',name))
                labels.append(os.path.join(data_dir,'anno','test',name.split('.')[0]+'.png'))
        self.labels = labels 
        self.images = images
        
    def __getitem__(self, index):
        #读取图像
        img_path,label_path =  self.images[index],self.labels[index]
        imgs_id = img_path.split("\\")[-1].split(".")[0]
        imgs = cv2.imread(img_path)
        imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
        lbls = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)
        #数据增强
        transform = A.Compose([
                A.Normalize (mean=[0.4754358, 0.35509014, 0.282971],std=[0.16318515, 0.15616792, 0.15164918]),
                ToTensorV2(),
            ])
        transformed = transform(image=imgs, mask=lbls)
        imgs = transformed['image']
        lbls = transformed['mask'].long()
        return imgs,lbls,imgs_id
    
    def __len__(self):
        return len(self.images)

def training_func(model: torch.nn.Module, optimizers: torch.optim.Optimizer,
                  criterion: torch.nn.CrossEntropyLoss(),
                  lr_schedulers: StepLR | None = None, max_steps: int | None = None,
                  max_epochs: int | None = None, *args, **kwargs):
    model.train()

    # prepare data
    imagenet_train_data = train_Dataset(data_dir=r"D:\31890\Desktop\codefile\data\Train_data\mseg_test_data",input_size=[512,512])
    train_dataloader = DataLoader(imagenet_train_data, batch_size=4, shuffle=True)

    #############################################################################
    # NNI may change the training duration by setting max_steps or max_epochs.
    # To ensure that NNI has the ability to control the training duration,
    # please add max_steps and max_epochs as constraints to the training loop.
    #############################################################################
    total_epochs = max_epochs if max_epochs else 20
    total_steps = max_steps if max_steps else 1000000
    current_steps = 0

    # training loop
    for _ in range(total_epochs):
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizers.zero_grad()
            pread = model(inputs)
            loss = criterion(pread, labels)
            loss.backward()
            optimizers.step()
            ######################################################################
            # stop the training loop when reach the total_steps
            ######################################################################
            current_steps += 1
            if total_steps and current_steps == total_steps:
                return
        lr_schedulers.step()


def evaluating_func(model: torch.nn.Module):
    model.eval()

    # prepare data
    imagenet_val_data = test_Dataset(r'D:\31890\Desktop\codefile\data\Train_data\mseg_test_data') 
    val_dataloader = DataLoader(imagenet_val_data, batch_size=1, shuffle=False)

    # testing loop
    correct = 0
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)
            correct += preds.eq(labels.view_as(preds)).sum().item()
    return correct / len(imagenet_val_data)


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_epochs=1
    total_steps =3
    cooldown_steps = 1


    model = segformer_m()
    optimizer = nni.trace(AdamW)(model.parameters(), lr=3e-5,eps=1e-8)
    criterion = torch.nn.CrossEntropyLoss()
    lr_scheduler = nni.trace(StepLR)(optimizer, step_size=5, gamma=0.1)
    dummy_input = torch.rand(1, 3, 224, 224).to(device)

    evaluator = TorchEvaluator(training_func=training_func, optimizers=optimizer, criterion=criterion,
                           lr_schedulers=lr_scheduler, dummy_input=dummy_input, evaluating_func=evaluating_func)
    config_list = [{
        'op_types': ['Conv2d'],
        'op_partial_names': ['encoder.stages.layer.{}.blocks'.format(i) for i in range(4)],
        'sparsity': 0.1
    }]

    # for name, layer in model.named_parameters(recurse=True):
    #     print(name, layer.shape, sep=" ")

    pruner = MovementPruner(model=model,
                        config_list=config_list,
                        evaluator=evaluator,
                        training_epochs=total_epochs,
                        training_steps=total_steps,
                        warm_up_step=1,
                        cool_down_beginning_step=total_steps - cooldown_steps,
                        regular_scale=10,
                        movement_mode='soft',
                        sparse_granularity='auto')
    
    _, attention_masks = pruner.compress()
    pruner.show_pruned_weights()

    # print(model)
    # y = model(dummy_input)
    # print(y.shape)


    # from nni.compression.pytorch.pruning import L1NormPruner
    # pruner = L1NormPruner(model, config_list)


    # # compress the model and generate the masks
    # _, masks = pruner.compress()
    # # show the masks sparsity
    # for name, mask in masks.items():
    #     print(name, ' sparsity : ', '{:.2}'.format(mask['weight'].sum() / mask['weight'].numel()))


    # # need to unwarp the model, if the model is wrawpped before speedup
    # pruner._unwrap_model()
    
    # # speedup the model
    # model.eval()

    # from nni.compression.pytorch.speedup import ModelSpeedup

    # ModelSpeedup(model, dummy_input.to(device), masks).speedup_model()