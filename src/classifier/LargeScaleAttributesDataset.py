from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import os
from skimage import io, transform
from PIL import Image
import numpy as np
import torch

# 078017
class LargeScaleAttributesDataset(Dataset):
    def __init__(self, attributes_file, attributes_list, label_list, root_dir, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.attributes = pd.read_csv(attributes_file, header=None)
        self.attributes = self.attributes[:-600]
        self.attributes[2] = self.attributes[2].map(lambda x: [int(y) for y in x.strip(" []").split()])
        self.num_attributes = len(self.attributes.iloc[0, 2])
        # print(self.num_attributes)
        self.attributes_names = pd.read_csv(attributes_list, header=None)
        self.label_list = pd.read_csv(label_list, header=None)
        self.root_dir = root_dir

        self.transform = transform

    def __len__(self):
        return len(self.attributes.index)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.attributes.iloc[idx, 1].strip())
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'attributes': torch.Tensor(self.attributes.iloc[idx, 2][:337])}

        return sample

if __name__ == "__main__":
    data = LargeScaleAttributesDataset( attributes_file='../../data/largescale/LAD_annotations/attributes.txt',
                                        attributes_list='../../data/largescale/LAD_annotations/attribute_list.txt',
                                        label_list='../../data/largescale/LAD_annotations/label_list.txt',
                                        root_dir='../../data/largescale/')
    print(len(data))
    print(len(data[0]['attributes']))
