import os
import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pretrainedmodels as pm
import pretrainedmodels.utils as utils
from torchsummary import summary

class AttributeClassifier(nn.Module):
    def __init__(self, out_features=10, base_model_name='xception', hidden_size=1024):
        super(AttributeClassifier, self).__init__()
        self.bm = pm.__dict__[base_model_name](num_classes=1000, pretrained='imagenet')
        self.bm_feature_dim = self.bm.last_linear.in_features
        self.input_size = self.bm.input_size
        self.bm.last_linear = pm.utils.Identity()

        for bm_param in self.bm.parameters():
            bm_param.requires_grad = False
        
        self.fc1 = nn.Linear(self.bm_feature_dim, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_features)

        torch.nn.init.kaiming_uniform_(self.fc1.weight.data, nonlinearity='relu')
        torch.nn.init.xavier_uniform_(self.fc2.weight.data)

    def forward(self, input):
        z = self.bm(input)
        z = self.fc1(z)
        z = self.bn1(z)
        z = F.relu(z)
        logits = self.fc2(z)
        return logits

if __name__ == "__main__":
    model = AttributeClassifier(10)
    summary(model, input_size=(3, 299, 299))