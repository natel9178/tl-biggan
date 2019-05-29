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

dataset_root = '../../data/largescale/'
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])

def run_epoch(model, data, device, is_train=False, optimizer=None, calculate_every=100):
    if is_train: model.train()
    else: model.eval()

    total_loss = 0
    n_total = 0
    n_total_correct = 0
    iterations = 0
    total_f1 = 0
    calc_iterations = 0

    batch_iterator = tqdm(data, mininterval=2, desc='  - (Training)   ', leave=False)
    with torch.set_grad_enabled(is_train):
        for batch in batch_iterator:
            if is_train: optimizer.zero_grad()
            
            x, y = batch['image'].to(device), batch['attributes'].to(device)
            preds = model(x)
            do_calculations = (iterations % calculate_every == 0) or not is_train
            loss, n_correct, total_labels, f1 = u.calculate_performance(preds, y, do_calculations=do_calculations)

            if is_train:
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            iterations += 1
            if do_calculations:
                n_total_correct += n_correct
                n_total += total_labels
                total_f1 += f1
                calc_iterations += 1

            batch_iterator.set_description( 'accuracy: {accu:3.3f}%, avg_loss: {avg_loss:8.5f}, avg_f1: {f1:3.3f}'.format(accu=100*n_total_correct/n_total,
                avg_loss=total_loss/iterations, f1=total_f1/calc_iterations))
    
    return total_loss, n_total_correct / n_total, iterations, total_f1 / calc_iterations


def train(model, training_data, validation_data, optimizer, scheduler, device, opt, start_epoch=0, log_tensorboard=None):
    metrics = []
    if log_tensorboard:
        writer = SummaryWriter(log_tensorboard)

    for epoch_i in range(start_epoch, opt.epoch):
        print('[ Epoch', epoch_i, ']')

        #- Pass through training data
        start = time.time()

        current_lr = scheduler.get_lr()[0]
        print('    - [Info] current LR is', current_lr)

        train_loss, train_accu, train_count, train_avg_f1 = run_epoch(model, training_data, device, is_train=True, optimizer=optimizer)
        print('  - (Training) accuracy: {accu:3.3f} %, '\
              'avg_loss: {loss:8.5f}, avg_f1: {f1:3.3f}, elapse: {elapse:3.3f} min'.format(accu=100*train_accu,
                  loss=train_loss/train_count, f1=train_avg_f1, elapse=(time.time()-start)/60))

        #- Pass through validation data
        start = time.time()
        valid_loss, valid_accu, valid_count, valid_avg_f1 = run_epoch(model, validation_data, device, is_train=False)
        print('  - (Validation) accuracy: {accu:3.3f} %, '\
                'avg_loss: {loss:8.5f}, avg_f1: {f1:3.3f}, elapse: {elapse:3.3f} min'.format(
                    accu=100*valid_accu, loss=valid_loss/valid_count, f1=valid_avg_f1, elapse=(time.time()-start)/60))

        metrics += [valid_avg_f1]

        scheduler.step(metrics=valid_loss) # Valid_loss

        #- Save checkpoint
        if opt.save_model:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'settings': opt,
                'epoch': epoch_i}

            if opt.save_mode == 'all':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                if valid_avg_f1 >= max(metrics):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')
        
        if log_tensorboard:
            writer.add_scalar('data/training_average_loss', train_loss/train_count, epoch_i)
            writer.add_scalar('data/training_accuracy', train_accu, epoch_i)
            writer.add_scalar('data/training_average_f1', train_avg_f1, epoch_i)
            writer.add_scalar('data/validation_average_loss', valid_loss/valid_count, epoch_i)
            writer.add_scalar('data/validation_accuracy', valid_accu, epoch_i)
            writer.add_scalar('data/validation_average_f1', valid_avg_f1, epoch_i)
            writer.add_scalar('data/lr', current_lr, epoch_i)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epoch', type=int, default=100)
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-lr', type=float, default=1e-3)

    parser.add_argument('-load_model', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-log_tensorboard', default=None)

    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    device = torch.device('cuda' if opt.cuda else 'cpu')

    model = AttributeClassifier(out_features=359, device=device)
    u.unfreeze_layers(model)

    tf = transforms.Compose([ transforms.RandomResizedCrop(229), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize ])
    full_dataset = LargeScaleAttributesDataset( attributes_file=os.path.join(dataset_root, 'LAD_annotations/attributes.txt'),
                                                attributes_list=os.path.join(dataset_root, 'LAD_annotations/attribute_list.txt'),
                                                label_list= os.path.join(dataset_root, 'LAD_annotations/label_list.txt'),
                                                root_dir=dataset_root,
                                                transform=tf)
    training_data, validation_data = u.create_dataloaders(full_dataset, batch_size=opt.batch_size)

    #- Output total number of parameters
    summary(model, input_size=(3, 299, 299))

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
    # scheduler_plateau = ls.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    scheduler_cosine = ls.CosineAnnealingLR(optimizer, len(training_data)*100, 0)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=4, total_epoch=10, after_scheduler=scheduler_cosine)

    start_epoch = 0
    log_tensorboard = opt.log_tensorboard
    if opt.load_model:
        checkpoint = torch.load(opt.load_model)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        saved_opt = checkpoint['settings']
        if 'optimizer' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer'])
        if not log_tensorboard: log_tensorboard = saved_opt.log_tensorboard

    train(model, training_data, validation_data, optimizer, scheduler_warmup, device, opt, start_epoch, log_tensorboard)

if __name__ == "__main__":
    main()