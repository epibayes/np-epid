import torch
import numpy as np
import pandas as pd
from .simulator import Simulator
from torch.distributions import Normal

class CRKPTransmissionSimulator(Simulator):
    def __init__(self, path, prior_mu, prior_sigma, n_sample=None):
        self.n_sample = n_sample
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.d_x = 1

        if n_sample is not None:
            self.data, self.theta = self.simulate_data()
        # who is present when?
        self.W = pd.read_csv(f"{path}/facility_trace.csv", index_col=0).values
        # tests upon entry

        self.N, self.T = self.W.shape

        self.V = pd.read_csv(f"{path}/screening.csv", index_col=0).values
        # observed infections count
        # consider spinning off into separate function
        self.x_o = self.load_observed_data(path)

    def load_observed_data(self, path):
        X = pd.read_csv(f"{path}/infections.csv", index_col=0).sum(0).values
        return torch.tensor(X).unsqueeze(0) / self.N

    def get_observed_data(self):
        return self.x_o
    
    def simulate_data(self):
        logbetas = self.sample_logbeta(self.n_sample, seed=5)
        xs = torch.empty((self.n_sample, self.d_x))
        for i in range(self.n_sample):
            rs = 5 * i
            xs[i] = self.CRKP_simulator(
                np.array(logbetas[i]), rs
                ).flatten()

        return xs, logbetas.float()
    
    def CRKP_simulator(self, logbeta, seed=None):
        if seed is not None:
            np.random.seed(seed)
        beta = np.exp(logbeta)
        N, T = self.W.shape
        beta = np.exp(logbeta)
        # current admitted status
        w = np.zeros(N)
        X = np.empty((N, T))
        # current status
        x = np.empty(N); x[:] = np.nan
        # # current infection status
        # I = np.zeros(N)
        
        for t in range(T):
            # case 1: not present
            # if absent, set to nan
            # otherwise, inherit old status
            X[:, t] = np.where(1 - self.W[:, t], np.nan, x)
            # case 2: new arrival
            newly_admitted = self.W[:, t] * (1 - w)
            # if newly admitted, load test data if available, otherwise default to last status
            # will this under-report? if someone gets tested a day after arrival
            X[:, t] = np.where(newly_admitted, self.V[:, t], X[:, t])
            # ALTERNATIVELY
            # inherit infection statuses from ground truth
            # case 3: already admitted and susceptible
            # randomly model transmission event
            # otherwise, inherit old status
            staying = self.W[:, t] * w
            # who is infected?
            I = (x == 1).astype(int)
            hazard = I.sum() * beta * np.ones(N)
            p = 1 - np.exp(-hazard / N) # not the end of the world to normalize by size of population
            X[:, t] = np.where(staying * (1 - I), np.random.binomial(1, p, N), X[:, t])
            x = X[:, t]
            w = self.W[:, t]

        data = np.nansum(X, axis=0) / N # scaling may speed up training
        return torch.tensor(data).float()

    def sample_logbeta(self, N, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        return torch.normal(self.prior_mu, self.prior_sigma, (N, 1))
    





# consider taking a rolling mean/sum by week as the relevant observed data
# do we really care about faithfully capturing daily transmission?

