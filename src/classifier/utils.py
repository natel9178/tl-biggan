
import torch.utils.data as d
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from losses import functional_bfe_with_logits
from sklearn.metrics import f1_score


def calculate_performance(pred, gold, do_calculations=True):
    loss, number_actual_correct, total_labels, f1 = None, None, None, None

    loss = functional_bfe_with_logits(pred, gold)

    if do_calculations:
        probs = torch.sigmoid(pred)
        predictions = torch.round(probs)
        # print(predictions)
        # print(gold)
        n_correct = torch.sum(predictions == gold) # dim=0 for per attribute accuracy
        number_actual_correct = n_correct.item()
        total_labels = predictions.shape[0] * predictions.shape[1]
        f1 = f1_score(gold.cpu().detach().numpy(), predictions.cpu().detach().numpy(), average='weighted')
        # print(f1, n_correct.item(), predictions.shape[0] * predictions.shape[1])

    # TODO: Per attribute accuracy

    return loss, number_actual_correct, total_labels, f1

def unfreeze_layers(model):
    for bm_param in model.bm.block12.parameters():
        bm_param.requires_grad = True
    for bm_param in model.bm.conv3.parameters():
        bm_param.requires_grad = True
    for bm_param in model.bm.bn3.parameters():
        bm_param.requires_grad = True
    for bm_param in model.bm.conv4.parameters():
        bm_param.requires_grad = True
    for bm_param in model.bm.bn4.parameters():
        bm_param.requires_grad = True

def split_dataset(dataset, training_split=0.9):
    training_length = int(len(dataset) * training_split)
    valid_len = int((len(dataset) - training_length) / 2)
    test_len = int(len(dataset) - training_length - valid_len)
    training_dataset, validation_dataset, test_dataset = d.random_split(dataset, [training_length, valid_len, test_len])
    return training_dataset, validation_dataset, test_dataset

def create_dataloaders(training_dataset, validation_dataset, training_split=0.9, batch_size=2, overfit_len=None, validation_batch_size=128):
    training_data = d.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
    validation_data = d.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)

    return training_data, validation_data

out_features = 337