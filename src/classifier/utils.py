
import torch.utils.data as d
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch


def calculate_performance(pred, gold):
    probs = torch.sigmoid(pred)
    loss = F.binary_cross_entropy(probs, gold)
    
    predictions = torch.round(probs)
    n_correct = torch.sum(predictions == gold) # dim=0 for per attribute accuracy

    # TODO: Per attribute accuracy

    return loss, n_correct.item()

def create_dataloaders(dataset, training_split=0.9, batch_size=2, overfit_len=None):
    training_length = int(len(dataset) * training_split)
    if overfit_len:
        training_dataset, validation_dataset, _ = d.random_split(dataset, [2, 2, len(dataset) - 4])
    else:
        training_dataset, validation_dataset = d.random_split(dataset, [training_length, len(dataset) - training_length])

    training_data = d.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)
    validation_data = d.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)

    return training_data, validation_data