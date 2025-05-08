import torch
import numpy as np
import pandas as pd
from .simulator import Simulator
from ..utils import contact_matrix, categorical_sample
from torch.distributions import MultivariateNormal
from hydra.utils import get_original_cwd


class PhyloSimulator(Simulator):
    def __init__(self, beta_true, prior_mu, prior_sigma, observed_seed, 
                 time_first, n_sample=None, notebook_mode=False, log_scale=True,
                 flatten=False, load_data=False, epid_only=False):
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
        self.flatten = flatten
        self.epid_only = epid_only
        postfix = "_epid_only" if epid_only else ""
        if n_sample:
            simulate = True
            if load_data:
                try:
                    print("Loading training data...")
                    self.data = torch.load(f"{prefix}/sim_data/phylo_data_{n_sample}{postfix}.pt",
                                           weights_only=True)
                    self.theta = torch.load(f"{prefix}/sim_data/phylo_theta_{n_sample}{postfix}.pt",
                                            weights_only=True)
                    if log_scale: self.theta = torch.log(self.theta)
                    simulate = False
                except FileNotFoundError:
                    print("Saved training data not found!")
            if simulate:
                print("Simulating training data...")
                self.data, self.theta = self.simulate_data()
                print("Writing out simulated data...")
                torch.save(self.data, f"{prefix}/sim_data/phylo_data_{n_sample}{postfix}.pt")
                theta_out = torch.exp(self.theta) if log_scale else self.theta
                torch.save(theta_out, f"{prefix}/sim_data/phylo_theta_{n_sample}{postfix}.pt")
            
            # no need to embed if it's just epid data
            self.d_x  = self.data[0].shape[0] if self.epid_only else self.data[0].shape
            
            
    
    def simulate_data(self):
        thetas = self.sample_logbeta(self.n_sample).float()
        if not self.log_scale:
            thetas = np.exp(thetas)
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
        if self.log_scale:
            beta = np.exp(theta)
        else:
            beta = theta
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
            # who was infected at the last timestep?
            Ia = x[w > 0]
            # set up hazard matrix: [i, j] entry is contribution 
            # from patient j acting on patient i
            # facility effect
            hazard = Ia * beta[0] * np.ones(M) / capacity[0]
            hazard = np.tile(hazard, (M, 1))
            # floor effect
            fa = f[w > 0].astype(int) + 1
            fC = contact_matrix(fa)
            hazard += (fC * Ia) * beta[fa] / capacity[fa]
            # room effect
            ra = r[w > 0].astype(int)
            rC = contact_matrix(ra)
            infected_roommates = (rC * Ia)
            room_count[t-1] = (infected_roommates.sum(1) > 1).sum()
            hazard += infected_roommates * beta[-1] / capacity[-1]
            p = np.zeros(N)
            # reinsert M probabilities (for present patients)
            # into N total patient indices
            p[w > 0] = 1 - np.exp(-hazard.sum(1))
            # third case: already admitted and susceptible
            # simulate infection as a bernoulli random variable
            staying = self.W[:, t] * w
            X[:, t] = np.where(staying * (x == 0), np.random.binomial(1, p, N), X[:, t])
            if self.epid_only:
                continue
            # compute cluster assignment probabilities
            ka = k[w > 0].astype(int)
            kI = np.eye(N_k+1)[ka].T[1:] # ignore contribution from "zero cluster"
            # this is calculating the individual hazard contribution from each donor
            cluster_scores = kI @ hazard.T
            # normalize scores to probabilities
            with np.errstate(divide="ignore", invalid="ignore"):
                cluster_probs = cluster_scores / cluster_scores.sum(0)
            cluster_assignments = categorical_sample(cluster_probs)
            kb = np.zeros(N)
            kb[w > 0] = cluster_assignments 
            # new infection? assign cluster
            K[:, t] = np.where((X[:, t] == 1) * (x == 0), kb, K[:, t])
            # if X is positive, cluster assignment should be nonzero
            # cluster overlap by room
            assert K[:, t][X[:, t] == 1].min() > 0
            roommate_clusters = rC * ka
            for rk in roommate_clusters:
                nonzero_ix = np.nonzero(rk)
                unique, counts = np.unique(rk[nonzero_ix], return_counts=True)
                for u, c in zip(unique, counts):
                    if c > 1: # there are roommates with the same cluster!
                        room_cluster[u-1, t] += 1
        
            
        w = self.W[:, T-1]
        ra = self.R[:, T-1][w > 0]
        rC = contact_matrix(ra)
        Ia = X[:, T-1][w > 0]
        infected_roommates = (rC * Ia).sum(1)
        room_count[T-1] = (infected_roommates > 1).sum()
        if not self.epid_only:
            # compute last room cluster overlap
            ka = K[:, T-1][w > 0].astype(int)
            roommate_clusters = rC * ka
            for rk in roommate_clusters:
                nonzero_ix = np.nonzero(rk)
                unique, counts = np.unique(rk[nonzero_ix], return_counts=True)
                for u, c in zip(unique, counts):
                    if c > 1:
                        room_cluster[u-1, T-1] += 1
        
        
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
        if self.epid_only:
            return torch.from_numpy(demog).flatten()
        # genomic data
        cluster_counts = np.zeros((N_k, T))
        cluster_floor_counts = np.zeros((self.N_f, N_k, T))
        for t in range(T):
            for n in range(N):
                x = X[n, t]
                k = K[n, t]
                f = self.F[n, t]
                if k and not np.isnan(k):
                    cluster_counts[int(k)-1, t] += 1
                    cluster_floor_counts[int(f), int(k)-1, t] += 1
                    
        genom = np.concatenate([
            cluster_counts, 
            cluster_floor_counts.reshape(-1, T),
            room_cluster
            ])
    
        output = np.concatenate([demog, genom])
        if self.time_first:
            output = output.T
        return torch.from_numpy(output)
        
        
                    
        
                        
        