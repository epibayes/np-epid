import torch
import lightning as L
from torch.utils.data import DataLoader, random_split
import yaml
import numpy as np

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
    # special case for non-batched inputs
    else:
        L = torch.zeros(dim, dim, device=values.device)
        tril_ix = torch.tril_indices(dim, dim)
        L[tril_ix[0], tril_ix[1]] = values[0]
    return L

def diag(values):
    if values.shape[0] > 1:
        L = torch.diag_embed(values)
    # special case for non-batched inputs
    else:
        L = torch.diag(values[0])
    return L

def contact_matrix(arr):
    x, y = np.meshgrid(arr, arr)
    return (x == y).astype(int)

def save_results(posterior_params, val_losses, cfg,
                 multidim):
    if multidim:
        mu = posterior_params[0].tolist()
        L = posterior_params[1]
        sigma = (L @ L.T).tolist()
        sdiag = (L @ L.T).diag().tolist()
        lognormal_mean = np.exp(
            np.array(mu) + np.array(sdiag) / 2
        )
        print(np.round(lognormal_mean, 3))
        print(np.round(mu, 3))
        print(np.round(sdiag, 3))
    else:
        mu = posterior_params[0].item()
        sigma = posterior_params[1].item()
        print(np.round(np.exp(mu + sigma**2 / 2), 3))
        print(np.round(mu, 3))
        print(np.round(sigma, 3))
    results = {"mu": mu, "sigma":sigma,
               "val_loss": val_losses[-1],
               "n_sample": cfg[cfg.experiment]["n_sample"],
               "seed": cfg[cfg.experiment]["observed_seed"],
               "batch_size": cfg["train"]["batch_size"]}
    for key in cfg["model"]:
        results[key] = cfg["model"][key]
    # should probably save seed, etc.
    with open("results.yaml", "w", encoding="utf-8") as yaml_file:
        yaml.dump(results, yaml_file)