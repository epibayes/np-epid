import torch
import numpy as np
from torch.utils.data import Dataset

class TestDataset(Dataset):
    # can the density network learn a simple gaussian?
    def __init__(self, n_sample, random_state):
        self.n_sample = n_sample
        self.x, self.theta = self.simulate_data(random_state)
        self.d_x = 1
        self.d_theta = 1

    def __len__(self):
        return self.n_sample

    def __getitem__(self, index):
        return self.x[index], self.theta[index]
    
    def simulate_data(self, random_state):
        np.random.seed(random_state)
        theta = np.random.normal(0, 1, self.n_sample)
        x = np.random.normal(0, 3, self.n_sample)
        return torch.tensor(x).float(), torch.tensor(theta).float()
    
    def get_observed_data(self):
        np.random.seed(4)
        return torch.normal(0,1, size=(32, 1))
    
    def evaluate(self, mu, sigma):
        mu_mse = (mu ** 2).mean().item()
        sigma_mse = ((sigma - 1)**2).mean().item()
        print(f"mu, MSE: {mu_mse:.3f}")
        print(f"sigma, MSE: {sigma_mse:.3f}")



class ExponentialToyDataset(Dataset):
    def __init__(self, n_obs, n_sample, random_state, shape, scale, summary=True):
        self.n_obs = n_obs
        self.n_sample = n_sample
        self.shape = shape
        self.scale = scale
        # use a summary statistic to "help"
        self.summary = summary

        self.x, self.theta = self.simulate_data(random_state)

        self.d_theta = 1
        self.d_x = 1 if summary else n_obs
        self.theta_true = 10

    def __len__(self):
        return self.n_sample

    def __getitem__(self, index):
        return self.x[index], self.theta[index]
    

    def simulate_data(self, random_state):
        # sample from prior
        np.random.seed(random_state)
        lamda = np.random.gamma(self.shape, self.scale, self.n_sample)
        theta = 1/lamda
        # matrix of shape (n_samples, n_observations)
        theta = np.tile(theta, (self.n_obs, 1)).T
        x = np.random.exponential(theta)
        # collapse theta back down
        theta = theta[:, 0]
        if self.summary:
            x = x.mean(1)
        return torch.tensor(x).float(), torch.tensor(theta).float()
    
    def get_observed_data(self):
        np.random.seed(4)
        observed_data = np.random.exponential(self.theta_true, 500)
        observed_data = torch.tensor(observed_data).float()
        return observed_data.mean() if self.summary else observed_data
    
        


