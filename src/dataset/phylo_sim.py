import torch
import numpy as np
import pandas as pd
from .simulator import Simulator
from ..utils import contact_matrix, categorical_sample
from torch.distributions import MultivariateNormal
from hydra.utils import get_original_cwd
import os


class PhyloSimulator(Simulator):
    def __init__(self, beta_true, prior_mu, prior_sigma, observed_seed, n_sample,
                 notebook_mode=False):
        self.n_sample = n_sample
        prefix = ".." if notebook_mode else get_original_cwd()
        self.W = pd.read_csv(f"{prefix}/sim_data/facility_trace.csv").values
        self.V = pd.read_csv(f"{prefix}/sim_data/screening.csv").values
        self.F = pd.read_csv(f"{prefix}/sim_data/floor_trace.csv").values
        self.R = pd.read_csv(f"{prefix}/sim_data/room_trace.csv").values
        self.cluster_indices = pd.read_csv(f"{prefix}/sim_data/cluster_lookup.csv").values[:, 0]
        self.L = np.zeros(self.W.shape[0])
        for i, k in enumerate(self.cluster_indices):
            self.L[k] = i+1
        # self.n_clusters = len(self.clusters)
        
        self.beta_true = beta_true
        
        self.prior_mu = torch.full((7,), prior_mu).float()
        self.prior_sigma = torch.diag(torch.full((7,), prior_sigma)).float()
        self.obs = observed_seed
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
    
    def get_observed_data(self):
        theta_true = np.log(np.array(self.beta_true))
        # TODO: log scale switching
        x_o = self.simulate(theta_true, self.obs)
        return x_o.float()
        
    
    def simulate(self, theta, seed):
        beta = np.exp(theta)
        
        np.random.seed(seed)
        M = 100 # don't hardcode this
        N, T = self.W.shape
        N_k = int(self.L.max())
        # infection statuses
        X = np.empty((N, T))
        X[:, 0] = self.V[:, 0]
        # cluster assignments
        K = np.empty((N, T))
        # importation cases start a cluster index
        K[:, 0] = np.where(self.V[:, 0] == 1, self.L, np.nan)
        # negative admits are assigned no cluster
        K[:, 0] = np.where(self.V[:, 0] == 0, 0, K[:,0])
   
        capacity = np.array([M] + [M/5]*5 + [M/50])
        room_count = np.empty(T)
        for t in range(1, T):
            x = X[:, t-1]
            k = K[:, t-1]
            w = self.W[:, t-1]
            f = self.F[:, t-1]
            r = self.R[:, t-1]
            # if absent, set to nan. Otherwise, default to last status
            X[:, t] = np.where(1 - self.W[:, t], np.nan, x)
            K[:, t] = np.where(1 - self.W[:, t], np.nan, k)
            newly_admitted = self.W[:, t] * (1 - w)
            assert (~np.isnan(self.V[:, t]) == newly_admitted).all()
            # if newly admitted, load test data if available, otherwise default to last status
            X[:, t] = np.where(newly_admitted, self.V[:, t], X[:, t])
            # if there is a newly screened case, add a new cluster
            # otherwise, assign "no cluster" to a negative admission
            # K[:, t] = np.where(self.V[:, t] == 1, self.L, K[:, t])
            # K[:, t] = np.where(self.V[:, t] == 0, 0, K[:, t])
            K[:, t] = np.where(newly_admitted, self.L, K[:, t])
            # case 3: already admitted and susceptible
            # randomly model transmission event
            # otherwise, inherit old status
            staying = self.W[:, t] * w
            # who was infected at the last timestep?
            Ia = x[w > 0] # 
            hazard = Ia * beta[0] * np.ones(M) / capacity[0]
            hazard = np.tile(hazard, (M, 1))
            fa = f[w > 0].astype(int)
            fC = contact_matrix(fa)
            # guarantee that there are no infecteds who aren't present
            # how many infected floormates?
            hazard += (fC * Ia) * beta[fa] / capacity[fa]
            ra = r[w > 0].astype(int)
            rC = contact_matrix(ra)
            infected_roommates = (rC * Ia)
            room_count[t-1] = (infected_roommates > 1).sum()
            hazard += infected_roommates * beta[-1] / capacity[-1]
            # new approach: keep hazard as an NxN matrix
            p = np.zeros(N)
            p[self.W[:, t] > 0] = 1 - np.exp(-hazard.sum(1))
            # this may be the wrong approach..
            # p = 0.2 
            X[:, t] = np.where(staying * (x == 0), np.random.binomial(1, p, N), X[:, t])
            
            # compute cluster assignment probabilities
            ka = k[w > 0].astype(int)
            kI = np.eye(N_k+1)[ka].T[1:] # ignore contribution from "zero cluster"
            
            # produce an N x N array such that summed up it's equal to the hazard..
            # this is calculating the individual hazard contribution from each donor
            cluster_scores = kI @ hazard.T
            # normalize scores to probabilities
            with np.errstate(divide="ignore", invalid="ignore"):
                cluster_probs = cluster_scores / cluster_scores.sum(0)
                cluster_probs = np.where(np.isnan(cluster_probs), 1, cluster_probs)
                
            
            cluster_assignments = categorical_sample(cluster_probs)
            kb = np.zeros(N)
            
            kb[self.W[:, t] > 0] = cluster_assignments + 1 
            K[:, t] = np.where((X[:, t] == 1) * (x == 0), kb, K[:, t])
            
            