import torch
import lightning as L
from torch.utils.data import DataLoader, Dataset, random_split
import yaml
import numpy as np
import pandas as pd
import glob
from sklearn.datasets import make_moons
from omegaconf import OmegaConf

PROBLEM_KEYS = ["beta_true"]

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
        self.val_size = val_size

    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val, self.val_size)
    
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


def categorical_sample(p):
    p = np.nan_to_num(p)
    K, M = p.shape
    sample = np.empty(M)
    # this could possibly be optimized
    for m in range(M):
        probs = p[:, m]
        # freaking floating point error.
        if probs.sum() == 0:
            sample[m] = 0
        else:
            sample[m] = np.random.choice(K, p=probs) + 1
    
    return sample

def save_results(posterior_params, val_losses, cfg, name):
    results = {"val_loss": val_losses[-1]}
    if posterior_params:
        # special case for homogeneous models
        if name in ["si-model", "crkp"]:
            mu = posterior_params[0].item()
            sigma = posterior_params[1].item()
            print(np.round(mu, 3))
            print(np.round(sigma, 3))
        else:
            mu = posterior_params[0].tolist()
            L = posterior_params[1]
            sigma = (L @ L.T).tolist()
            sdiag = (L @ L.T).diag().tolist()
            print(np.round(mu, 3))
            print(np.round(sdiag, 3)) # marginal variances
        results["mu"] = mu
        results["sigma"] = sigma
    for key in cfg["simulator"]:
        results[key] = cfg["simulator"][key]
    for key in cfg["model"]:
        results[key] = cfg["model"][key]
    # well this is irritating
    for key in PROBLEM_KEYS:
        if key in results:
            try:
                results[key] = OmegaConf.to_object(results[key])
            except ValueError:
                pass
    # should probably save seed, etc.
    with open("results.yaml", "w", encoding="utf-8") as yaml_file:
        yaml.dump(results, yaml_file)
        
# reading multiruns

def get_results(path, multirun=True):
    extension =  "/results.yaml"
    if multirun: extension = "/**" + extension
    # if multirun:
    #     extension = "/**/results.yaml"
    # else:
    #     extension = "/results.yaml"
    results = glob.glob(path + extension)
    data = dict()
    for res in results:
        with open(res, "r") as stream:
            yml = yaml.safe_load(stream)
            for k, v in yml.items():
                if k not in data.keys():
                    data[k] = [v]
                else:
                    data[k].append(v)
    data = pd.DataFrame(data)
    # in practice, i don't tune these hyperparameters
    for c in ["_target_", "lr", "batch_size", "dropout", "seed"]:
        try:
            data.drop(columns = c, inplace=True)
        except KeyError:
            print(f"Missing column {c}")
    return data
        
# LIKELIHOOD BASED ESTIMATION

def simulator(alpha, beta, gamma, N, T, seed, het=False):
    if not het:
        beta = [beta]
    X  = np.empty((N, T))
    np.random.seed(seed)
    X[:, 0] = np.random.binomial(1, alpha, N)
    F = np.arange(N) % 5
    R = np.arange(N) % (N // 2)
    fC = contact_matrix(F)
    rC = contact_matrix(R)
    for t in range(1, T):
        I = X[:, t-1]
        # components dependent on individual covariates
        hazard = compute_hazard(beta, I, N, F, fC, rC, het)
        p = 1 - np.exp(-hazard)
        new_infections = np.random.binomial(1, p, N)
        X[:, t] = np.where(I, np.ones(N), new_infections)
        discharge = np.random.binomial(1, gamma, N)
        screening = np.random.binomial(1, alpha, N)
        X[:, t] = np.where(discharge, screening, X[:, t])
    return X

def compute_hazard(beta, I, N, F, fC, rC, het):
    hazard = I.sum() * beta[0] * np.ones(N) / N
    if het:
        hazard += (fC * I).sum(1) * beta[F+1] / 60
        hazard += (rC * I).sum(1) * beta[-1] / 2
    return hazard

def nll(beta, alpha, gamma, N, T, X, het):
    # beta = beta / np.array([1, 300, 300, 300, 300, 300, 300])
    return - x_loglikelihood(beta, alpha, gamma, N, T, X, het)

def x_loglikelihood(logbeta, alpha, gamma, N, T, X, het=False):
    ans = np.log(
        alpha ** X[:, 0] * (1 - alpha) ** (1 - X[:, 0])
        ).sum()
    beta = np.exp(logbeta)
    if not het:
        beta = [beta]
    F = np.arange(N) % 5
    R = np.arange(N) % (N // 2)
    fC = contact_matrix(F)
    rC = contact_matrix(R)
    for t in range(1, T):
        xs = X[:, t-1]
        xt = X[:, t]
        hazard = compute_hazard(beta, xs, N, F, fC, rC, het)
        ans += (xt * xs  * np.log(
            gamma * alpha + (1 - gamma)
        )).sum()
        ans += (xt * (1 - xs)  * np.log(
            gamma * alpha + (1 - gamma) * (1 - np.exp(- hazard))
        )).sum()
        ans += ((1 - xt) * xs  * np.log(
            gamma * (1 - alpha) + 1e-8
        )).sum()
        ans += ((1 - xt) * (1 - xs) * np.log(
            gamma *(1 - alpha) + (1 - gamma) * (np.exp(- hazard))
        )).sum()
    return ans


### misc

def lognormal_sd(log_mean, log_sd):
    a = np.exp(log_sd**2) - 1
    b = np.exp(2*log_mean + log_sd**2)
    return (a*b)**0.5

class MoonsDataset(Dataset):
    def __init__(self, n_sample, random_state):
        self.n_sample = n_sample
        self.random_state = random_state
        self.data = self._make_data()

    def _make_data(self):
        arr = make_moons(self.n_sample, noise=0.05, random_state=self.random_state)[0]
        return torch.from_numpy(arr).float()

    def __len__(self):
        return self.n_sample
    
    def __getitem__(self, index):
        return torch.empty(0), self.data[index]
