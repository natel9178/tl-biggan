import os
import math
import time
import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as ls
import torch.utils.data as d
from AttributeClassifier import AttributeClassifier
from LargeScaleAttributesDataset import LargeScaleAttributesDataset
import utils as u
from torchvision import transforms
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from scheduler import GradualWarmupScheduler
import pickle

dataset_root = '../../data/largescale/'
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])

def main():
    tf = transforms.Compose([ transforms.RandomResizedCrop(229), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize ])
    full_dataset = LargeScaleAttributesDataset( attributes_file=os.path.join(dataset_root, 'LAD_annotations/attributes.txt'),
                                                attributes_list=os.path.join(dataset_root, 'LAD_annotations/attribute_list.txt'),
                                                label_list= os.path.join(dataset_root, 'LAD_annotations/label_list.txt'),
                                                root_dir=dataset_root,
                                                transform=tf)
    training_data, validation_data, test_data = u.split_dataset(full_dataset, 0.9)
    pickle.dump( training_data, open( os.path.join(dataset_root,"train.pkl"), "wb" ) )
    pickle.dump( validation_data, open( os.path.join(dataset_root,"val.pkl"), "wb" ) )
    pickle.dump( test_data, open( os.path.join(dataset_root,"test.pkl"), "wb" ) )

if __name__ == "__main__":
    main()