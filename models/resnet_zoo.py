# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 15:24:51 2018

resnet loader from pytorch model zoo

ref: https://github.com/pytorch/examples/pull/58

@author: Γιώργος
"""

import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def resnet_base_loader(pretrained, version):
    assert version in ['18','34','50','101','152']
    if version=='18':
        model_ft = resnet18loader(pretrained)
    elif version=='34':
        model_ft = resnet34loader(pretrained)
    elif version=='50':
        model_ft = resnet50loader(pretrained)
    elif version=='101':
        model_ft = resnet101loader(pretrained)
    elif version=='152':
        model_ft = resnet152loader(pretrained)
    else:
        print('Should never be here')
    
    return model_ft
    
def resnet_loader(num_classes, dropout, pretrained, feature_extraction, version, channels, pad_input):
    model_ft = resnet_base_loader(pretrained, version)    
    set_parameter_requires_grad(model_ft, feature_extraction)

    if pad_input:
        modules = []
        modules.append(nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=(96,0), bias=False))
        if channels != 3:
            prev_conv = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else: # if still on RGB keep the pretrained weights for the layer
            prev_conv = model_ft.conv1
        modules.append(prev_conv)
        model_ft.conv1 = nn.Sequential(*modules)
    else:
        if channels != 3:
            model_ft.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Dropout(p=dropout),
                                nn.Linear(num_ftrs, num_classes))    
    return model_ft

def location_resnet_loader(num_classes, dropout, pretrained, feature_extraction, version):
    model_ft = resnet_base_loader(pretrained, version)
    
    set_parameter_requires_grad(model_ft, feature_extraction)
    
    model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(40,240), stride=2, padding=3,
                               bias=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Dropout(p=dropout),
                                nn.Linear(num_ftrs, num_classes))
    
    return model_ft
    

def resnet18loader(pretrained):
    model_ft = models.resnet18(pretrained=pretrained)
    return model_ft

def resnet34loader(pretrained):
    model_ft = models.resnet34(pretrained=pretrained)
    return model_ft

def resnet50loader(pretrained):
    model_ft = models.resnet50(pretrained=pretrained)
    return model_ft

def resnet101loader(pretrained):
    model_ft = models.resnet101(pretrained=pretrained)
    return model_ft

def resnet152loader(num_classes, dropout, pretrained, feature_extraction, channels):
    model_ft = models.resnet152(pretrained=pretrained)
    return model_ft