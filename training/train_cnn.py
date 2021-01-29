#!/usr/bin/env python3
# coding: utf-8
import glob 
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from ccsne_models import cnn, utils

import argparse

def parse_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-label')
    parser.add_argument('--data-dir')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--ifos')
    parser.add_argument('--n-classes', type=int)
    params = parser.parse_args()
    return params

params = parse_cmd()
run_label = params.run_label
data_dir = params.data_dir
seed = params.seed
ifos = params.ifos
n_classes = params.n_classes
out_dir = f'/home/ethan.marx/ccsne/training/train_output/{run_label}/'

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

logfile = open(out_dir + 'training_log.txt', 'w')
logfile.write(f'data seed: {seed}\n')
logfile.write(f'ifos: {ifos}\n')
logfile.write('epoch train_loss val_loss accuracy\n')

# training constants
num_workers = 64
batch_size = 256 
max_epochs = 150

lr = 1e-4
weight_decay = 1e-3
device = torch.device('cuda')

# generate data loaders
train_data, val_data, test_data = utils.gen_datasets(data_dir, seed=seed, ifos=ifos)
train_dl = DataLoader(train_data, batch_size = batch_size, shuffle=True, pin_memory=False, num_workers=num_workers)
val_dl = DataLoader(val_data, batch_size = batch_size, shuffle=True, pin_memory=False, num_workers=num_workers)
torch.save(test_data, out_dir + 'test_data')

# initiate model
if ifos == 'H1L1':
    model = cnn.CCSNeMultiIFO(n_classes)
else:
    model = cnn.CCSNeSingleIFO(n_classes)

model.to(device) 
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)
#scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.1)

for epoch in range(max_epochs):
    # set training loss to 0
    train_loss = 0
    model.train()
    # iterate over train dl
    for data, labels in train_dl:
        # reset gradients
        optimizer.zero_grad()
       	 
        if type(model) == cnn.CCSNeMultiIFO:
            # squeeze data
			
            data = data.squeeze(1)
            labels = labels.squeeze(1)
        
        elif type(model) == cnn.CCSNeSingleIFO:
            # unsqueeze data
            data = data.unsqueeze(1)
            labels = labels.squeeze(1)
        
        # move data to GPU
        data = data.float().to(device)
        
        labels = labels.long().to(device)
        
        # evaluate model on train data
        outputs = model(data)
       	 
        # evaluate loss and update total
        loss = criterion(outputs, labels)
        
        # backprop to calculate gradients and optimize
        loss.backward()
        optimizer.step()
                         
        # update loss
        train_loss += loss.item()
        
    # adjust learning rate
    #scheduler.step()             
    # now evaluate model on vaidation set
    
    # initiate validation loss
    val_loss = 0
    
    accuracies = np.array([])
    model.eval()
    
    # don't wan't to update model params for validation
    with torch.no_grad():
        for data, labels in val_dl:
            
            if type(model) == cnn.CCSNeMultiIFO:
                # squeeze data
                data = data.squeeze(1)
                labels = labels.squeeze(1)
        
            elif type(model) == cnn.CCSNeSingleIFO:
                # unsqueeze data
                data = data.unsqueeze(1)
                labels = labels.squeeze(1)
      
			# move data to GPU
            data = data.float().to(device)
            labels = labels.long().to(device)
            
            # evaluate model in val data
            outputs = model(data)
            
            # calculate loss
            loss = criterion(outputs, labels)
            
            # update val loss
            val_loss += loss.item()
            
            
            # calculate accuracy
            _, predicted = torch.max(outputs, 1)
            accuracy = torch.tensor(torch.sum(predicted == labels).item() / len(predicted))
            accuracies = np.append(accuracies, accuracy)

    accuracy = np.mean(accuracies)
    
    # save summary of epoch
    
    
    
    logfile.write(f'{epoch} {train_loss} {val_loss} {accuracy}\n')
    print(f'{epoch} {train_loss} {val_loss} {accuracy}\n')     
    # save model
    if not os.path.exists(out_dir + f'/models/'):
        os.mkdir(out_dir + f'/models/')
    torch.save(model.state_dict(), out_dir +  f'/models/epoch_{epoch}')
    
logfile.close()
