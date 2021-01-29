#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.utils.data as utils
import os

import time

import argparse

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.init as init
import sys

from ccsne_models import autoencoder, utils 


def parse_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-label')
    parser.add_argument('--data-dir')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--ifos')
    parser.add_argument('--class-num', type=int)
    params = parser.parse_args()
    return params


params = parse_cmd()
run_label = params.run_label
data_dir = params.data_dir
seed = params.seed
ifos = params.ifos
class_num = params.class_num
out_dir = f'/home/ethan.marx/ccsne/training/train_output/{run_label}/'

batch_size = 256
num_workers = 128

n_data = 200000  # each class has 200000 samples in data dir
partition = (70,15,15)
indices = np.random.permutation(n_data)
indices = indices + n_data*(class_num)
n_train = int(partition[0]* n_data / 100)
n_val = int(partition[1]* n_data / 100)
n_test = int(partition[2]* n_data / 100)

indices_train = indices[:n_train]
indices_val = indices[n_train : n_train + n_val]
indices_test = indices[n_train + n_val:]
ifos = 'H1'
train_dataset = utils.dataset(indices_train, data_dir, ifos=ifos)
val_dataset = utils.dataset(indices_val, data_dir, ifos = ifos)
test_dataset = utils.dataset(indices_test, data_dir, ifos = ifos)
torch.save(test_dataset, out_dir + 'test_data')


train_dl = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, pin_memory=False, num_workers=num_workers)
val_dl = DataLoader(val_dataset, batch_size = batch_size, shuffle=True, pin_memory=False, num_workers=num_workers)
test_dl = DataLoader(val_dataset, batch_size = batch_size, shuffle=True, pin_memory=False, num_workers=num_workers)
  


beta=1
N_FLOWS = 10
Z_DIM = 200

n_steps = 0

device = torch.device('cuda')

model = autoencoder.VAE_NF(N_FLOWS, Z_DIM, device=device).to(device)

lr = 1e-4
weight_decay = 1e-3

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, 20, 0.1)

def train():
    global n_steps
    train_loss = []
    model.train()
    tot_loss = 0
    for batch_idx, data in enumerate(train_dl):
        x, _ = data
        
        start_time = time.time()
        x = x.unsqueeze(1)
        x = x.float().to(device)

        x_tilde, kl_div = model(x)
        x_tilde = x_tilde.unsqueeze(1)
        
       	mseloss = nn.MSELoss(reduction='sum')
        
        loss_recons = mseloss(x_tilde, x) / x.size(0)
        
        loss = loss_recons + beta * kl_div

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append([loss_recons.item(), kl_div.item()])
        tot_loss += loss
      
    tot_loss /= len(train_dataset)
    scheduler.step()
    return tot_loss

def evaluate():
 
    start_time = time.time()
    val_loss = 0
    model.eval()
    
    with torch.no_grad():
        for batch_idx, data in enumerate(val_dl):
            x, _ = data
            x = x.unsqueeze(1)
            x = x.float().to(device)

           
            x_tilde, kl_div = model(x)
            x_tilde = x_tilde.unsqueeze(1)
            mseloss = nn.MSELoss(reduction='sum')
           
            loss_recons = mseloss(x_tilde,x ) / x.size(0)
          
            loss = loss_recons + beta * kl_div

            val_loss += loss
           
    val_loss /= len(val_dataset)
    
    return val_loss


max_epochs = 200
LAST_SAVED = -1
PATIENCE_COUNT = 0
PATIENCE_LIMIT = 5
BEST_LOSS = np.inf
train_losses = np.array([])
val_losses = np.array([])
for epoch in range(max_epochs):
    print("Epoch {}:".format(epoch))
    train_loss = train()
    val_loss = evaluate()
    train_losses = np.append(train_losses, train_loss)
    val_losses = np.append(val_losses, train_loss)
    
    np.savetxt(out_dir + '/train_loss.txt', train_losses)
    np.savetxt(out_dir + '/val_loss', val_losses)
    print(f'Training and evaluation Completed! Validation Loss: {val_loss:5.4f} Training Loss: {train_loss:5.4f}')

    if val_loss <= BEST_LOSS:
        PATIENCE_COUNT = 0
        BEST_LOSS = val_loss
        LAST_SAVED = epoch
        print("Saving model!")
        if not os.path.exists(out_dir + '/models/'):
            os.mkdir(out_dir + '/models/')
        torch.save(model.state_dict(), out_dir + f'/models/epoch_{epoch}')
    
    else:
        PATIENCE_COUNT += 1
        print(f"Not saving model! Last saved: {LAST_SAVED}")
        if PATIENCE_COUNT > 100:
            print("Patience Limit Reached")
            break
