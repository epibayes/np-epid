import torch
import numpy as np
import pandas as pd
from .simulator import Simulator
from ..utils import contact_matrix, categorical_sample
from torch.distributions import MultivariateNormal
from hydra.utils import get_original_cwd
import os


class PhyloSimulator(Simulator):
    def __init__(self, beta_true, prior_mu, prior_sigma, observed_seed, 
                 time_first, n_sample=None, notebook_mode=False, log_scale=True):
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
        self.log_scale = log_scale
        self.d_theta = 7
        self.N_f = 5 # number of floors
        self.N_r = 50 # number of rooms
        self.name = "phylo-sim"
        self.time_first = time_first
        self.d_x = None
        self.n_sample = n_sample
        if n_sample is not None:
            self.data, self.theta = self.simulate_data()
            self.d_x  = self.data[0].shape # here goes nothing
        
    
    def simulate_data(self):
        thetas = self.sample_logbeta(self.n_sample).float()
        xs = []
        for i in range(self.n_sample):
            seed = 3*i
            xs.append(
                self.simulate(
                np.array(thetas[i]), seed
                )
            )
        xs = torch.stack(xs)
        return xs.float(), thetas
    
    def sample_logbeta(self, N):
        torch.manual_seed(4)
        mvn = MultivariateNormal(self.prior_mu, self.prior_sigma)
        return mvn.sample((N,))
    
    def get_observed_data(self):
        theta_true = np.array(self.beta_true)
        if self.log_scale:
            theta_true = np.log(theta_true)
        x_o = self.simulate(theta_true, self.obs)
        if self.n_sample is not None:
            x_o = x_o.unsqueeze(0)
        return x_o.float()
        
    
    def simulate(self, theta, seed):
        beta = np.exp(theta)
        
        np.random.seed(seed)
        M = 100
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
   
        capacity = np.array([M] + [M/self.N_f]*self.N_f + [M/self.N_r])
        room_count = np.empty(T)
        
        room_cluster = np.zeros((N_k, T))
        
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
            room_count[t-1] = (infected_roommates.sum(1) > 1).sum()
            hazard += infected_roommates * beta[-1] / capacity[-1]
            # new approach: keep hazard as an NxN matrix
            p = np.zeros(N)
            p[self.W[:, t] > 0] = 1 - np.exp(-hazard.sum(1))
            # this may be the wrong approach..
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
            
            # cluster overlap by room
            roommate_clusters = rC * ka
            for rk in roommate_clusters:
                nonzero_ix = np.nonzero(rk)
                unique, counts = np.unique(rk[nonzero_ix], return_counts=True)
                for u, c in zip(unique, counts):
                    if c > 1: # multiple patients of the same cluster type
                        room_cluster[u-1, t] = c
            
            
        w = self.W[:, T-1]
        ra = self.R[:, T-1][w > 0]
        rC = contact_matrix(ra)
        Ia = X[:, T-1][w > 0]
        infected_roommates = (rC * Ia).sum(1)
        room_count[T-1] = (infected_roommates > 1).sum()
        # compute last room cluster overlap
        ka = K[:, T-1][w > 0].astype(int)
        roommate_clusters = rC * ka
        for rk in roommate_clusters:
            nonzero_ix = np.nonzero(rk)
            unique, counts = np.unique(rk[nonzero_ix], return_counts=True)
            for u, c in zip(unique, counts):
                if c > 1:
                    room_cluster[u-1, T-1] = c
        
        
        # start writing out data
        total_count = np.nansum(X, axis=0)
        # demographic data
        demog  = [total_count]
        for i in range(self.N_f):
            # does this work with matrix indexing?
            floor_count = np.nansum(X * (self.F == i), axis=0)
            demog.append(floor_count)
        demog.append(room_count)
        demog = np.stack(demog)
        # genomic data
        cluster_counts = np.zeros((N_k, T))
        cluster_floor_counts = np.zeros((N_k, self.N_f, T))
        for t in range(T):
            for n in range(N):
                k = K[n, t]
                f = self.F[n, t]
                if k and not np.isnan(k):
                    cluster_counts[int(k)-1, t] += 1
                    cluster_floor_counts[int(k)-1, int(f), t] += 1
                    
        genom = np.concatenate([
            cluster_counts, 
            cluster_floor_counts.reshape(-1, T),
            room_cluster
            ])
    
        output = np.concatenate([demog, genom])
        if self.time_first:
            output = output.T
        return torch.from_numpy(output)
        
        
                    
        
                        
        