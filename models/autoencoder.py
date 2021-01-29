import math
import numpy as np
import scipy as sp
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

class FCNN(nn.Module):
    """
    Code taken from Sangeon Park - https://github.com/SangeonPark/QUASAR/
    Simple fully connected neural network.
    """
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.network(x)


class MAF(nn.Module):
    """
    Code taken from Sangeon Park - https://github.com/SangeonPark/QUASAR/

    Masked auto-regressive flow.
    [Papamakarios et al. 2018]
    """
    def __init__(self, dim, hidden_dim=8, base_network=FCNN, device=torch.device('cpu')):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        self.device = device
        self.initial_param = nn.Parameter(torch.Tensor(2))
        for i in range(1, dim):
            self.layers += [base_network(i, 2, hidden_dim)]
        self.reset_parameters()

    def reset_parameters(self):
        if torch.device != torch.device('cpu'):
            init.uniform_(self.initial_param, -math.sqrt(0.5), math.sqrt(0.5)).to(self.device)
        else:
            init.uniform_(self.initial_param, -math.sqrt(0.5), math.sqrt(0.5))
    
    def forward(self, x):
        z = torch.zeros_like(x)
        if torch.device != torch.device('cpu'):
            log_det = torch.zeros(z.shape[0]).to(self.device)
        else:
            log_det = torch.zeros(z.shape[0])
        for i in range(self.dim):
            if i == 0:
                mu, alpha = self.initial_param[0], self.initial_param[1]
            else:
                out = self.layers[i - 1](x[:, :i])
                mu, alpha = out[:, 0], out[:, 1]
            z[:, i] = (x[:, i] - mu) / torch.exp(alpha)
            log_det -= alpha
        return z.flip(dims=(1,)), log_det

    def inverse(self, z):
        x = torch.zeros_like(z)
        if GPU == True:
            log_det = torch.zeros(z.shape[0]).to(self.device)
        else:
            log_det = torch.zeros(z.shape[0])
        z = z.flip(dims=(1,))
        for i in range(self.dim):
            if i == 0:
                mu, alpha = self.initial_param[0], self.initial_param[1]
            else:
                out = self.layers[i - 1](x[:, :i])
                mu, alpha = out[:, 0], out[:, 1]
            x[:, i] = mu + torch.exp(alpha) * z[:, i]
            log_det += alpha
        return x, log_det


class VAE_NF(nn.Module):
    def __init__(self, K, D, device=torch.device('cpu')):
        super().__init__()
        self.dim = D
        self.K = K
        self.device = device
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, stride=3, padding=3),
            nn.ReLU(True),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, stride=3, padding=3), 
            nn.ReLU(True),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=3, padding=3), 
            nn.ReLU(True),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=3, padding=3),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=5, padding=3),
            nn.Flatten(),
            nn.Linear(2048, self.dim * 2)
        )
        
        self.to_decoder = nn.Linear(self.dim, 2048)
     
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=5, stride=5, padding=3, output_padding=3),
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=5, stride=3, padding=3, output_padding=2),
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=5, stride=3, padding=3, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=5, stride=3, padding=3, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=8, out_channels=1, kernel_size=5, stride=3, padding=3, output_padding=1),
            nn.ReLU(True),
            nn.Flatten(),
            #nn.Linear(1728, 6144),
        )
        
        flow_init = MAF(dim=D, device=self.device)
        flows_init = [flow_init for _ in range(K)]
        
        if self.device != torch.device('cpu'):
            prior = MultivariateNormal(torch.zeros(D).to(self.device), torch.eye(D).to(self.device))
        else:
            prior = MultivariateNormal(torch.zeros(D), torch.eye(D))
        self.flows = NormalizingFlowModel(prior, flows_init, device=self.device)
        
    def forward(self, x):
        
        # run encoder
        enc = self.encoder(x)
        
        
        # get NF params
        mu = enc[:, :self.dim]
        log_var = enc[:, self.dim: self.dim * 2]

        # Re-parametrize
        sigma = (log_var * .5).exp()
        z = mu + sigma * torch.randn_like(sigma)
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Construct more expressive posterior with NF
        z_k, _, sum_ladj = self.flows(z)
        z_k = z_k.unsqueeze(1)
        kl_div = kl_div / x.size(0) - sum_ladj.mean()  # mean over batch
        #print(np.shape(z_k))
        
        # prepare for decoder
        x_prime =self.to_decoder(z_k)
        x_prime = x_prime.view(x_prime.shape[0], 128, -1)
        
        
        x_prime = self.decoder(x_prime)
       
        
     
        return x_prime, kl_div


class NormalizingFlowModel(nn.Module):
    '''
    Code taken from Sangeon Park - https://github.com/SangeonPark/QUASAR/	
    '''
    def __init__(self, prior, flows, device=torch.device('cpu')):
        super().__init__()
        self.prior = prior
        self.device = device
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        m, _ = x.shape
        if torch.device != torch.device('cpu'):
            log_det = torch.zeros(m).to(self.device)
        else:
            log_det = torch.zeros(m)
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
        z, prior_logprob = x, self.prior.log_prob(x)
        return z, prior_logprob, log_det

    def inverse(self, z):
        m, _ = z.shape
        if torch.device != torch.device('cpu'):
            log_det = torch.zeros(m).to(self.device)
        else:
            log_det = torch.zeros(m)
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z)
            log_det += ld
        x = z
        return x, log_det

    def sample(self, n_samples):
        z = self.prior.sample((n_samples,))
        x, _ = self.inverse(z)
        return x
    
    
