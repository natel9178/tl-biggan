
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

def create_dataloaders(dataset, training_split=0.9, batch_size=2):
    training_length = int(len(dataset) * training_split)
    training_dataset, validation_dataset, _ = d.random_split(dataset, [16, 6, len(dataset) - 16 - 6])#[training_length, len(dataset) - training_length])

    training_data = d.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)
    validation_data = d.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)

    return training_data, validation_data