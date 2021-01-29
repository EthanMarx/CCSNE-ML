import torch.nn as nn
import numpy as np

class CCSNeSingleIFO(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self._n_classes = n_classes
        self.layers = nn.Sequential(
            nn.Conv1d(1, 16, 3),
            nn.MaxPool1d(4, 4),
            nn.ReLU(inplace=True), 
            nn.Conv1d(16, 32, 3), 
            nn.MaxPool1d(4,4), 
            nn.ReLU(inplace=True), 
            nn.Conv1d(32, 64, 3), 
            nn.MaxPool1d(4,4), 
            nn.ReLU(inplace=True),
            
            )

        self.fc = nn.Sequential(
            nn.Linear(6080, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.BatchNorm1d(32),
            nn.Linear(32, self._n_classes),
            )

    def forward(self, x):
        x = self.layers(x)
        # flatten
        x =  x.view(x.size(0), -1)
       	#print(np.shape(x)) 
        # classify
        x = self.fc(x)
        return x

class CCSNeMultiIFO(nn.Module):
    def __init__(self, n_classes):
        
        super().__init__()
        self._n_classes = n_classes
        self.layers = nn.Sequential(
            nn.Conv1d(2, 16, 3),
            nn.MaxPool1d(4, 4),
            nn.ReLU(inplace=True), 
            nn.Conv1d(16, 32, 3), 
            nn.MaxPool1d(4,4), 
            nn.ReLU(inplace=True), 
            nn.Conv1d(32, 64, 3), 
            nn.MaxPool1d(4,4), 
            nn.ReLU(inplace=True),
          
            )

        self.fc = nn.Sequential(
            nn.Linear(6080, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.BatchNorm1d(32),
            nn.Linear(32, self._n_classes),
            )

    def forward(self, x):
        x = self.layers(x)
        # flatten
        x =  x.view(x.size(0), -1)
       	#print(np.shape(x)) 
        # classify
        x = self.fc(x)
        return x

  
