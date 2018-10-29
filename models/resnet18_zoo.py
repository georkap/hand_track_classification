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

def resnet18loader(num_classes, pretrained, feature_extraction):
    model_ft = models.resnet18(pretrained=pretrained)
    set_parameter_requires_grad(model_ft, feature_extraction)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))

    return model_ft

def resnet50loader(num_classes, pretrained, feature_extraction):
    model_ft = models.resnet50(pretrained=pretrained)
    set_parameter_requires_grad(model_ft, feature_extraction)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))

    return model_ft

def resnet101loader(num_classes, pretrained, feature_extraction):
    model_ft = models.resnet101(pretrained=pretrained)
    set_parameter_requires_grad(model_ft, feature_extraction)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))

    return model_ft