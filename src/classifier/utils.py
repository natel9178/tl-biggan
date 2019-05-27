
import torch.utils.data as d
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from losses import functional_bfe_with_logits
from sklearn.metrics import f1_score


def calculate_performance(pred, gold):
    loss = functional_bfe_with_logits(pred, gold)

    probs = torch.sigmoid(pred)
    predictions = torch.round(probs)
    # print(predictions)
    # print(gold)
    n_correct = torch.sum(predictions == gold) # dim=0 for per attribute accuracy
    total_labels = predictions.shape[0] * predictions.shape[1]
    f1 = f1_score(gold.cpu().detach().numpy(), predictions.cpu().detach().numpy(), average='weighted')

    # print(f1, n_correct.item(), predictions.shape[0] * predictions.shape[1])

    # TODO: Per attribute accuracy

    return loss, n_correct.item(), total_labels, f1


def create_dataloaders(dataset, training_split=0.9, batch_size=2, overfit_len=None, validation_batch_size=128):
    training_length = int(len(dataset) * training_split)
    if overfit_len:
        training_dataset, validation_dataset, _ = d.random_split(dataset, [overfit_len, overfit_len, len(dataset) - overfit_len*2])
    else:
        training_dataset, validation_dataset = d.random_split(dataset, [training_length, len(dataset) - training_length])

    training_data = d.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)
    validation_data = d.DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True, drop_last=False, num_workers=2)

    return training_data, validation_data