
import torch.utils.data as d
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

def functional_bfe_with_logits(input, target, gamma=2., alpha=0.25, size_average=True, eps=1e-10):
    probs = torch.clamp(torch.sigmoid(input), eps, 1.0 - eps)
    loss = - (( alpha * ( (1 - probs) ** gamma ) * target * torch.log(probs))  + ((1. - alpha) * (1. - target) * (probs ** gamma) * torch.log(1 - probs)))
    if size_average: return loss.mean()
    else: return loss.sum()

class BinaryFocalLossWithLogits(nn.Module):
    def __init__(self, gamma=2., alpha=0.25, size_average=True, eps=1e-10): # alpha < 0.5 deweights negative classes
        super(BinaryFocalLossWithLogits, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.eps = eps

    def forward(self, input, target):
        return functional_bfe_with_logits(input, target, self.gamma, self.alpha, self.size_average, self.eps)

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()