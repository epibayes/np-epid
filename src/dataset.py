import torch
import numpy as np
from torch.utils.data import Dataset

class ExponentialToyDataset(Dataset):
    def __init__(self, n_obs, n_sample, random_state):
        self.n_obs = n_obs
        self.n_sample = n_sample
        self.theta, self.x = self.simulate_data(random_state)

    def __len__(self):
        return self.n_sample

    def __getitem__(self, index):
        return self.x[index], self.theta[index]
    

    def simulate_data(self, random_state):
        # sample from prior
        np.random.seed(random_state)
        theta = np.random.gamma(0.1, 0.1, self.n_sample)
        theta = np.tile(theta, (self.n_obs, 1))
        x = np.random.exponential(1/theta)


