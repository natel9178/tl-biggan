import os
import math
import time
import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as d
from AttributeClassifier import AttributeClassifier
from LargeScaleAttributesDataset import LargeScaleAttributesDataset
import utils as u
from torchvision import transforms
from torchsummary import summary
# from torch.utils.tensorboard import SummaryWriter

dataset_root = '../../data/largescale/'
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])

def train_epoch(model, training_data, optimizer, device):
    model.train()

    total_loss = 0
    n_total = 0
    n_total_correct = 0

    for batch in tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):
        optimizer.zero_grad()
        x, y = batch['image'].to(device), batch['attributes'].to(device)

        preds = model(x)
        loss, n_correct = u.calculate_performance(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_total_correct += n_correct
        n_total += y.shape[0] * y.shape[1]
    
    return total_loss, n_total_correct / n_total


def eval_epoch(model, validation_data, device):
    model.eval()

    total_loss = 0
    n_total = 0
    n_total_correct = 0

    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc='  - (Validating)   ', leave=False):
            x, y = batch['image'].to(device), batch['attributes'].to(device)
            preds = model(x)
            loss, n_correct = u.calculate_performance(preds, y)

            total_loss += loss.item()
            n_total_correct += n_correct
            n_total += y.shape[0] * y.shape[1]

    return total_loss, n_total_correct / n_total


def train(model, training_data, validation_data, optimizer, device, opt):
    valid_accus = []
    # if opt.log_tensorboard:
    #     writer = SummaryWriter()

    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        #- Pass through training data
        start = time.time()
        train_loss, train_accu = train_epoch(model, training_data, optimizer, device)
        print('  - (Training) accuracy: {accu:3.3f} %, '\
              'loss: {loss:8.5f}, elapse: {elapse:3.3f} min'.format(accu=100*train_accu,
                  loss=train_loss, elapse=(time.time()-start)/60))

        #- Pass through validation data
        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, device)
        print('  - (Validation) accuracy: {accu:3.3f} %, '\
                'loss: {loss:8.5f}, elapse: {elapse:3.3f} min'.format(
                    accu=100*valid_accu, loss=valid_loss, elapse=(time.time()-start)/60))

        valid_accus += [valid_accu]

        #- Save checkpoint
        if opt.save_model:
            model_state_dict = model.state_dict()
            checkpoint = {
                'model': model_state_dict,
                'settings': opt,
                'epoch': epoch_i}

            if opt.save_mode == 'all':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')
        
        # if opt.log_tensorboard:
        #     writer.add_scalar('data/training_loss', train_loss, epoch_i)
        #     writer.add_scalar('data/training_accuracy', train_accu, epoch_i)
        #     writer.add_scalar('data/validation_loss', valid_loss, epoch_i)
        #     writer.add_scalar('data/validation_accuracy', valid_accu, epoch_i)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epoch', type=int, default=100)
    parser.add_argument('-batch_size', type=int, default=8)
    parser.add_argument('-lr', type=float, default=1e-3)

    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-log_tensorboard', action='store_true')

    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    device = torch.device('cuda' if opt.cuda else 'cpu')

    model = AttributeClassifier(out_features=359, device=device)

    tf = transforms.Compose([ transforms.RandomCrop(229), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize ])
    full_dataset = LargeScaleAttributesDataset( attributes_file=os.path.join(dataset_root, 'LAD_annotations/attributes.txt'),
                                                attributes_list=os.path.join(dataset_root, 'LAD_annotations/attribute_list.txt'),
                                                label_list= os.path.join(dataset_root, 'LAD_annotations/label_list.txt'),
                                                root_dir=dataset_root,
                                                transform=tf)
    training_data, validation_data = u.create_dataloaders(full_dataset, batch_size=opt.batch_size)

    #- Output total number of parameters
    summary(model, input_size=(3, 299, 299))

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)

    train(model, training_data, validation_data, optimizer, device, opt)

if __name__ == "__main__":
    main()