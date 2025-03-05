import torch
import numpy as np
import pandas as pd
from .simulator import Simulator
from ..utils import contact_matrix
from torch.distributions import MultivariateNormal
from hydra.utils import get_original_cwd
import os


class PhyloSimulator(Simulator):
    def __init__(self, beta_true, prior_mu, prior_sigma, observed_seed, n_sample,
                 notebook_mode=False):
        self.n_sample = n_sample
        prefix = ".." if notebook_mode else get_original_cwd()
        self.W = pd.read_csv(f"{prefix}/sim_data/facility_trace.csv")
        self.V = pd.read_csv(f"{prefix}/sim_data/screening.csv")
        self.F = pd.read_csv(f"{prefix}/sim_data/floor_trace.csv")
        self.R = pd.read_csv(f"{prefix}/sim_data/room_trace.csv")
        self.clusters = pd.read_csv(f"{prefix}/sim_data/cluster_lookup.csv")
        
        self.prior_mu = torch.full((7,), prior_mu).float()
        self.prior_sigma = torch.diag(torch.full((7,), prior_sigma)).float()
        
        self.d_theta = 7
        self.name = "phylo-sim"
        
    
    def simulate_data(self):
        thetas = self.sample_logbeta(self.n_sample).float()
        
        for i in range(self.n_sample):
            seed = 3*i
            self.simulate(
                np.array(thetas[i]), seed
            )
        
        return None, thetas
    
    def sample_logbeta(self, N):
        generator = torch.Generator()
        generator.manual_seed(4)
        mvn = MultivariateNormal(self.prior_mu, self.prior_sigma)
        return mvn.sample((N,), generator=generator)
    
    def simulate(self, theta, seed):
        beta = np.exp(theta)
        
        np.random.seed(seed)
        
        N, T = self.W.shape
        X = np.empty((N, T))
        
        X[:, 0] = self.V[:, 0]
        
        capacity = [N] + [N/5]*5 + [N/50] # should I be dividing by the number of rooms?
        
        for t in range(1, T):
            x = X[:, t-1]
            w = self.W[:, t-1]
            f = self.F[:, t-1]
            r = self.R[:, t-1]
        
            X[:, t] = np.where(1 - self.W[:, t], np.nan, x)            # case 2: new arrival
            newly_admitted = self.W[:, t] * (1 - w)
            # if newly admitted, load test data if available, otherwise default to last status
            X[:, t] = np.where(newly_admitted, self.V[:, t], X[:, t])
            # case 3: already admitted and susceptible
            # randomly model transmission event
            # otherwise, inherit old status
            staying = self.W[:, t] * w
            # who was infected at the last timestep?
            Ia = x[w > 0]
            hazard = Ia.sum() * beta[0] * np.ones(N) / capacity
            fa = f[w > 0]
            fC = contact_matrix(fa)
            # guarantee that there are no infecteds who aren't present
            # how many infected floormates?
            if self.het:
                hazard[w > 0] += (fC * Ia).sum(1) * beta[fa] / capacity[fa]
            ra = r[w > 0]
            rC = contact_matrix(ra)
            infected_roommates = (rC * Ia).sum(1)
            room_count[t-1] = (infected_roommates > 1).sum()
            if self.het:
                hazard[w > 0] += infected_roommates * beta[-1] / capacity[-1]
            p = 1 - np.exp(-hazard) # not the end of the world to normalize by size of population
            X[:, t] = np.where(staying * (x == 0), np.random.binomial(1, p, N), X[:, t])