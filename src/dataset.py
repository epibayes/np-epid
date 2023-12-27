import torch
import numpy as np
from torch.utils.data import Dataset

class ExponentialToyDataset(Dataset):
    def __init__(self, n_obs, n_sample, random_state, shape, scale, summary=True):
        self.n_obs = n_obs
        self.n_sample = n_sample
        self.shape = shape
        self.scale = scale
        # use a summary statistic to "help"
        self.summary = summary

        self.theta, self.x = self.simulate_data(random_state)

        self.d_theta = 1
        self.d_x = 1 if summary else n_obs

    def __len__(self):
        return self.n_sample

    def __getitem__(self, index):
        return self.x[index], self.theta[index]
    

    def simulate_data(self, random_state):
        # sample from prior
        np.random.seed(random_state)
        theta = np.random.gamma(self.shape, self.scale, self.n_sample)
        # matrix of shape (n_samples, n_observations)
        theta = np.tile(theta, (self.n_obs, 1)).T
        x = np.random.exponential(theta)
        # collapse theta back down
        theta = theta[:, 0]
        if self.summary:
            x = x.mean(1)
        return torch.tensor(theta).float(), torch.tensor(x).float()


