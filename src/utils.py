import torch
import lightning as L
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset

# theoretically, variational dropout takes care of overfitting
class DataModule(L.LightningDataModule):
    def __init__(self, dataset, seed, batch_size, train_frac):
        super().__init__()
        self.dataset = dataset
        self.seed = seed
        self.batch_size = batch_size
        self.train_frac = train_frac

    
    def setup(self, stage):
        train_size = int(self.train_frac * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train, self.val = random_split(
                self.dataset,
                (train_size, val_size),
                torch.Generator().manual_seed(self.seed)
        )

    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val, self.batch_size)
    
def lower_tri(values, dim):
    if values.shape[0] > 1:
        L = torch.zeros(values.shape[0], dim, dim, device=values.device)
        tril_ix = torch.tril_indices(dim, dim)
        L[:, tril_ix[0], tril_ix[1]] = values
    else:
        L = torch.zeros(dim, dim, device=values.device)
        tril_ix = torch.tril_indices(dim, dim)
        L[tril_ix[0], tril_ix[1]] = values[0]
    return L


class Simulator(Dataset):
    def __init__(self, n_sample, random_state):
        self.n_sample = None
        self.data = None
        self.theta = None
        self.d_x = None
        self.d_theta = None

    def __len__(self):
        return self.n_sample
    
    def __getitem__(self, index):
        return self.data[index], self.theta[index]
    
    def simulate_data(self, random_state):
        raise NotImplementedError
    
    def get_observed_data(self):
        raise NotImplementedError
    
    def evaluate(self, mu, sigma, data):
        pass

