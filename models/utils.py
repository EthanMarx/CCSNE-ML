import numpy as np
import torch
import glob
import os
from torch.utils.data import Dataset
import h5py

class dataset(Dataset):
    
    def __init__(self, indices, data_dir, ifos):
        """
        Args:
            data_dir (string): Path to the data directory with .pt file
        """
        self._data_dir = data_dir
        self._indices = indices
        self._ifos = ifos 
        
        
    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        i = self._indices[idx]
        data = torch.load(os.path.join(self._data_dir, f'data_point_{i}.pt'))
        if self._ifos == 'L1' or self._ifos == 'H1':
            ts = torch.Tensor(data['ts'][self._ifos])
        else:
            ts = torch.Tensor(np.vstack((data['ts']['H1'], data['ts']['L1'])))
           
        label = torch.Tensor(data['label'])
        
        return ts, label
    
    
    def get_class_data(self, cl, samples_per_class, data_dir, ifos):
        '''
        
        Function that returns subset of dataset from specified class 
        
        cl (int): class label for which you wan't to gather data
        samples_per_class (int): number of samples in class

        assumes data_dir is structured so datapoints are ordered by class (e.g 1-200 are class 0, 201-400 class 2, etc.)

        '''
    
        # get all indices
        all_indices = self.get_indices()

        # filter for indices from specific class
        class_indices = all_indices[np.logical_and(all_indices > cl*samples_per_class, all_indices < (cl+1)*(samples_per_class))]

        data_set = dataset(class_indices, self._data_dir, ifos=self._ifos)
        return data_set

    def get_indices(self):
        return self._indices
      

    def get_labels(self):
        labels = np.array([])
        for i in range(len(self)):
            dp = self[i]
            label = dp[1]
            labels = np.append(labels, label)
        return labels
        
            
        
        
def gen_datasets(data_dir, seed=None, partition=(80, 10, 10), ifos=None):
    
    n_data = len(glob.glob(os.path.join(data_dir, 'data_point*'))) 
    
    n_train = int(partition[0]* n_data / 100)
    n_val = int(partition[1]* n_data / 100)
    n_test = int(partition[2]* n_data / 100)
    
    np.random.seed(seed)
    indices = np.random.permutation(n_data)
    indices_train = indices[:n_train]
    indices_val = indices[n_train : n_train + n_val]
    indices_test = indices[n_train + n_val:]
    
    train_dataset = dataset(indices_train, data_dir, ifos=ifos)
    val_dataset = dataset(indices_val, data_dir, ifos = ifos)
    test_dataset = dataset(indices_test, data_dir, ifos = ifos)
    return train_dataset, val_dataset, test_dataset




class h5_dataset(Dataset):
    '''
    Class that generates dataset compatible with pytorch dl's from h5 file
    '''
    
    def __init__(self, h5_file, ifos, label):
        """
        Args:
            h5_file (string): Path to the h5 file
            ifos (string): which ifos using data for 
            label: label of the corresponding class
        """
        
        self._file = h5_file
        self._ifos = ifos
        self._label = label
        self.data = {}
        for ifo in ifos:
            self.data[ifo] = h5py.File(self._file, 'r')[ifo]['whitened timeseries']
        
    def __len__(self):
        ifo = self._ifos[0]
        return len(self.data[ifo])

    def __getitem__(self, idx):
        
        
        ts = [self.data[ifo][idx] for ifo in self._ifos]

        data = torch.Tensor(np.vstack((ts)))
        return data, self._label
        
        
        
            
   
        
