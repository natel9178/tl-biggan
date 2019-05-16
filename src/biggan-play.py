import torch
from biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal)

import logging
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
logging.basicConfig(level=logging.INFO)

model = BigGAN.from_pretrained('biggan-deep-512')

# Prepare a input
truncation = 0.4
class_vector = one_hot_from_names(['bubble', 'cat', 'mouse'], batch_size=3)
noise_vector = truncated_noise_sample(truncation=truncation, batch_size=3)

# All in tensors
noise_vector = torch.from_numpy(noise_vector)
class_vector = torch.from_numpy(class_vector)

# If you have a GPU, put everything on cuda
# noise_vector = noise_vector.to('cuda')
# class_vector = class_vector.to('cuda')
# model.to('cuda')

# Generate an image
with torch.no_grad():
    output = model(noise_vector, class_vector, truncation)

# If you have a GPU put back on CPU
output = output.to('cpu')

save_as_images(output)