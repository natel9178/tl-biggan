import os
import math
import time
import argparse
import h5py
from tqdm import tqdm
import torch
import numpy as np

from biggan import (BigGAN, one_hot_from_names, one_hot_from_int, truncated_noise_sample,
                                       save_as_images, display_in_terminal, convert_to_images)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', type=int, default=5)
    parser.add_argument('-batch_size', type=int, default=2)
    parser.add_argument('-save_loc', default='./gan_samples.hdf5')
    parser.add_argument('-truncation', type=float, default=0.7)
    parser.add_argument('-imagenet_class', type=int, default=235)
    parser.add_argument('-no_cuda', action='store_true')
    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    device = torch.device('cuda' if opt.cuda else 'cpu')

    generator = BigGAN.from_pretrained('biggan-deep-512').to(device)

    with h5py.File(opt.save_loc, 'a') as f:
        class_vector = torch.from_numpy(one_hot_from_int(opt.imagenet_class, batch_size=opt.batch_size)).to(device)

        np_class_vector = class_vector[0].to('cpu').numpy()
        should_rewrite = len(f.keys()) != 3 or ('class_vector' in f.keys() and (f['class_vector'][:] != np_class_vector).any())
        print('should_rewrite:', should_rewrite)
        if should_rewrite:
            _ = f.create_dataset("class_vector", data=np_class_vector)

        for i in tqdm(range(opt.epochs)):
            noise_vector = torch.from_numpy(truncated_noise_sample(truncation=opt.truncation, batch_size=opt.batch_size)).to(device)

            with torch.no_grad():
                output = generator(noise_vector, class_vector, opt.truncation)
                output = output.permute(0, 2, 3, 1)
                output = (((output + 1) / 2.0) * 256 ).round().clamp(0, 255)
            
            if should_rewrite:
                _ = f.create_dataset("images", data=output.detach().to('cpu').numpy().astype(np.uint8), compression='gzip', compression_opts=9, chunks=True, maxshape=(None,None,None,None))
                _ = f.create_dataset("latent_vector", data=noise_vector.to('cpu').numpy(), compression='gzip', compression_opts=9, chunks=True, maxshape=(None,None))
                should_rewrite = False
            else:
                f["images"].resize((f["images"].shape[0] + output.shape[0]), axis = 0)
                f["latent_vector"].resize((f["latent_vector"].shape[0] + noise_vector.shape[0]), axis = 0)
                f["images"][-output.shape[0]:] = output
                f["latent_vector"][-noise_vector.shape[0]:] = noise_vector

            
    

if __name__ == "__main__":
    main()