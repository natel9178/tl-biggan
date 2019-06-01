import os
import math
import time
import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as d
from classifier.AttributeClassifier import AttributeClassifier
from classifier.LargeScaleAttributesDataset import LargeScaleAttributesDataset
import classifier.utils as u
from torchvision import transforms
from classifier.Predictor import AttributeClassifierInference
import h5py
from torch.utils.data import Dataset, DataLoader

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])

class GenGanSamplesDataset(Dataset):
    def __init__(self, samples_filepath, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.samples_filepath = samples_filepath 
        self.dataset = None
        with h5py.File(self.samples_filepath, 'r') as file:
            self.dataset_len = len(file["images"])
        self.transform = transform

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.samples_filepath, 'r')['images']

        image = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, idx


def process_samples(predictor, dataloader, N, device):
    predictor.eval()
    labels = np.zeros((N,359))
    for batch in tqdm(dataloader, desc='Processing Generated Images'):
        images, idxs = batch[0].to(device), batch[1].to(device)
        
        with torch.no_grad():
            pred = predictor(images)
        labels[idxs] = pred.cpu().numpy()
    return labels 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-samples_loc', default='./tlgan/gan_samples')
    parser.add_argument('-model_weight_loc', default='./focal_model_2_finetune_2c')
    parser.add_argument('-no_cuda', action='store_true')
    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    device = torch.device('cuda' if opt.cuda else 'cpu')
    
    tf = transforms.Compose([ transforms.ToPILImage(), transforms.Resize(229), transforms.ToTensor(), normalize ])
    data = GenGanSamplesDataset(opt.samples_loc + '.hdf5', transform=tf)
    dataloader = DataLoader(data, batch_size=opt.batch_size, num_workers=2)

    checkpoint = torch.load(opt.model_weight_loc + '.chkpt', map_location=device)
    classifier = AttributeClassifier(out_features=359, device=device)
    classifier.load_state_dict(checkpoint['model'], strict=False)
    predictor = AttributeClassifierInference(attribute_classifier=classifier, device=device)
    labels = process_samples(predictor, dataloader, len(data), device)
    
    with h5py.File(opt.samples_loc + '_labels.hdf5', 'w') as l:
        print('Writing Labels')
        l.create_dataset('labels', data=labels, compression='gzip', compression_opts=9, chunks=True)
        
if __name__ == "__main__":
    main()