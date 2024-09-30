import torch
import numpy as np
import pandas as pd
from .simulator import Simulator
from ..utils import contact_matrix
from torch.distributions import MultivariateNormal

class CRKPTransmissionSimulator(Simulator):
    def __init__(self, path, prior_mu, prior_sigma, n_sample=None,
                 observed_seed=None, heterogeneous=True,
                 flatten=False, N=False):
        self.n_sample = n_sample
        self.het = heterogeneous
        self.d_theta = 8 if self.het else 1 # six floors, facility and room level transmission rates
        self.set_prior(prior_mu, prior_sigma)

        # who is present when?
        self.W = pd.read_csv(f"{path}/facility_trace.csv", index_col=0).values
        # tests upon entry
        self.V = pd.read_csv(f"{path}/screening.csv", index_col=0).values
        self.F = pd.read_csv(f"{path}/floor_trace.csv", index_col=0).values
        self.R = pd.read_csv(f"{path}/room_trace.csv", index_col=0).values
        self.N, self.T = self.W.shape
        self.d_x = self.T * 8 if self.het else self.T
        self.flatten = flatten # only set this false for ABC comparison
        self.lam = None
        if n_sample is not None:
            self.data, self.theta = self.simulate_data()

        self.x_o = self.load_observed_data(path)

        

    def set_prior(self, mu, sigma):
        if self.het:
            assert len(mu) == self.d_theta
            assert len(sigma) == self.d_theta
            self.prior_mu = torch.tensor(mu).float()
            self.prior_sigma = torch.diag(torch.tensor(sigma)).float()
        else:
            self.prior_mu = mu
            self.prior_sigma = sigma

    def load_observed_data(self, path):
        # todo: add logic for switching between homogeneous/hetero
        with open(f"{path}/observed_data.npy", "rb") as f:
            x = np.load(f)
        if self.het:
            return torch.tensor(x)
        else:
            return torch.tensor(x[0,:]).unsqueeze(0)

    def get_observed_data(self):
        x_o = self.x_o
        if self.flatten:
            return x_o.float().flatten().unsqueeze(0)
        else:
            return x_o.float()
    
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
        beta_0 = beta[0] if self.het else beta
        N, T = self.W.shape
        beta = np.exp(logbeta)
        # current admitted status
        w = np.zeros(N)
        # current floor
        f = np.zeros(N).astype(int)
        # current room
        r = np.zeros(N).astype(int)
        X = np.empty((N, T))
        # current status
        x = np.empty(N); x[:] = np.nan
        # # current infection status
        # note that no new infections will be simulated in timestep 0
        room_count = np.empty(T)
        for t in range(T):
            # case 1: not present
            # if absent, set to nan
            # otherwise, inherit old status
            X[:, t] = np.where(1 - self.W[:, t], np.nan, x)
            # case 2: new arrival
            newly_admitted = self.W[:, t] * (1 - w)
            # if newly admitted, load test data if available, otherwise default to last status
            X[:, t] = np.where(newly_admitted, self.V[:, t], X[:, t])
            # case 3: already admitted and susceptible
            # randomly model transmission event
            # otherwise, inherit old status
            staying = self.W[:, t] * w
            # who was infected at the last timestep?
            I = (x == 1).astype(int)
            assert I[w == 0].sum() == 0
            hazard = I.sum() * beta_0 * np.ones(N)
            if self.het:
                n_admitted = w.sum()
                if n_admitted == 0:
                    pass
                else:
                    # generate contact matrix
                    fa = f[w > 0]
                    fC = contact_matrix(fa)
                    # guarantee that there are no infecteds who aren't present
                    # how many infected floormates?
                    Ia = I[w > 0]
                    hazard[w > 0] += (fC * Ia).sum(1) * beta[fa]
                    ra = r[w > 0]
                    rC = contact_matrix(ra)
                    infected_roommates = (rC * Ia).sum(1)
                    room_count[t] = (infected_roommates > 1).sum() / 2
                    hazard[w > 0] += infected_roommates * beta[-1]
                p = 1 - np.exp(-hazard / N) # not the end of the world to normalize by size of population
                X[:, t] = np.where(staying * (1 - I), np.random.binomial(1, p, N), X[:, t])
            x = X[:, t]
            w = self.W[:, t]
            f = self.F[:, t]
            r = self.R[:, t]

        total_count = np.nansum(X, axis=0)

        if self.het:
            floor_counts = []
            for i in range(1,7):
                # does this work with matrix indexing?
                floor_count = np.nansum(X * (self.F == i), axis=0)
                floor_counts.append(floor_count)
                stats = [total_count] + floor_counts + [room_count]
            data = torch.tensor(np.stack(stats)).float()
        else:
            data =  torch.tensor(total_count).float()

        # if self.flatten:
        #     data = data.flatten()

        return data

    def sample_logbeta(self, N, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        if self.het:
            mvn = MultivariateNormal(self.prior_mu, self.prior_sigma)
            logbeta = mvn.sample((N,))
        else:
            logbeta = torch.normal(self.prior_mu, self.prior_sigma, (N, 1))
        return logbeta

