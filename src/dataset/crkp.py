import torch
import numpy as np
import pandas as pd
from .simulator import Simulator
from ..utils import contact_matrix
from torch.distributions import MultivariateNormal

SCALE = [129., 28., 38., 35., 27., 17., 2]

class CRKPTransmissionSimulator(Simulator):
    def __init__(self, path, prior_mu, prior_sigma, n_sample=None,
                 heterogeneous=True, name=None,
                 flatten=False, N=None,
                 summarize=False, pi=None, return_case_count=False,
                 transformer=False, load_data=False, postfix=""):
        self.n_sample = n_sample
        self.het = heterogeneous
        self.cap = np.array(SCALE)
        self.d_theta = 7 if self.het else 1 # five floors, facility and room level transmission rates
        self.set_prior(prior_mu, prior_sigma)
        # who is present when?
        self.W = pd.read_csv(f"{path}/facility_trace.csv", index_col=0).values
        # tests upon entry
        self.V = pd.read_csv(f"{path}/screening.csv", index_col=0).values
        self.F = pd.read_csv(f"{path}/floor_trace.csv", index_col=0).values
        self.R = pd.read_csv(f"{path}/room_trace.csv", index_col=0).values
        self.N, self.T = self.W.shape
        self.L = np.repeat(np.array(SCALE)[:, None], self.T, axis=1)
        if pi is not None:
            if np.isscalar(pi):
                self.pi = np.array([pi])
            else:
                self.pi = np.array(pi)
        else:
            self.pi = None
        if summarize:
            if transformer:
                self.d_x = (self.d_theta, self.T)
            else:
                self.d_x = self.T * self.d_theta
        else:
            self.d_x = self.T
        self.flatten = flatten # set false for ABC
        self.return_case_count = return_case_count
        if n_sample:
            simulate = True
            if load_data:
                try:
                    print("Loading training data...")
                    self.data = torch.load(f"{path}/data_{n_sample}_{postfix}.pt",
                                           weights_only=True)
                    self.theta = torch.load(f"{path}/theta_{n_sample}_{postfix}.pt",
                                            weights_only=True)
                    simulate = False
                except FileNotFoundError:
                    print("Saved training data not found!")
            if simulate:
                print("Simulating training data...")
                self.data, self.theta = self.simulate_data(summarize)
                print("Writing out simulated data...")
                torch.save(self.data, f"{path}/data_{n_sample}_{postfix}.pt")
                torch.save(self.theta, f"{path}/theta_{n_sample}_{postfix}.pt")

        self.x_o = self.load_observed_data(path, summarize)
        self.name = name
        self.mask = self.W
        

    def set_prior(self, mu, sigma):
        if self.het:
            if type(mu) == list:
                self.prior_mu = torch.tensor(mu).float()
            else:
                self.prior_mu = torch.full((7,), mu).float()
            self.prior_sigma = torch.diag(torch.full((7,), sigma)).float()
        else:
            self.prior_mu = mu
            self.prior_sigma = sigma

    def load_observed_data(self, path, summarize):
        if summarize:
            with open(f"{path}/observed_data.npy", "rb") as f:
                x = np.load(f)
            # scale    
            x = x / self.L
            if self.het:
                return torch.tensor(x)
            else:
                return torch.tensor(x[0,:]).unsqueeze(0)
        else:
            x = pd.read_csv(f"{path}/infections.csv", index_col=0).values
            return torch.tensor(x).unsqueeze(0)


    def get_observed_data(self):
        obs = self.x_o
        if self.flatten and self.summarize:
            obs =  obs.flatten().unsqueeze(0)
            
        return obs.float()
    
    def simulate_data(self, summarize=False):
        logbetas = self.sample_logbeta(self.n_sample, seed=6)
        # if summarize:
        #     xs = torch.empty(self.n_sample, self.d_x)
        # else:
        #     xs = torch.empty(self.n_sample, self.N, self.T)
        xs = []
        for i in range(self.n_sample):
            rs = 5 * i
            sim = self.CRKP_simulator(
                np.array(logbetas[i]), rs, summarize=summarize
                )
            if summarize: sim = sim.flatten()
            xs.append(sim)
            # xs[i] = sim.flatten() if summarize else sim
        xs = torch.stack(xs).float()

        return xs, logbetas.float()
    
    def CRKP_simulator(self, logbeta, seed=None, summarize=False, show_full=False):
        if seed is not None:
            np.random.seed(seed)
        beta = np.exp(logbeta)
        if self.pi is not None:
            beta = beta * self.pi
        beta_0 = beta[0] if self.het else beta
        N, T = self.W.shape
        X = np.empty((N, T))
        # load screen data for first day
        X[:, 0] = self.V[:, 0]
        room_count = np.empty(T)
        for t in range(1, T):
            x = X[:, t-1]
            w = self.W[:, t-1]
            f = self.F[:, t-1]
            r = self.R[:, t-1]
            # case 1: not present
            # if absent, set to nan
            # otherwise, inherit old status
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
            hazard = Ia.sum() * beta_0 * np.ones(N) / self.cap[0]
            fa = f[w > 0]
            fC = contact_matrix(fa)
            # guarantee that there are no infecteds who aren't present
            # how many infected floormates?
            if self.het:
                hazard[w > 0] += (fC * Ia).sum(1) * beta[fa] / self.cap[fa]
            ra = r[w > 0]
            rC = contact_matrix(ra)
            infected_roommates = (rC * Ia).sum(1)
            room_count[t-1] = (infected_roommates > 1).sum()
            if self.het:
                hazard[w > 0] += infected_roommates * beta[-1] / self.cap[-1]
            p = 1 - np.exp(-hazard) # not the end of the world to normalize by size of population
            X[:, t] = np.where(staying * (x == 0), np.random.binomial(1, p, N), X[:, t])

        if summarize:
            
            # compute last entry of room count
            r = self.R[:, T-1]
            w = self.W[:, T-1]
            x = X[:, T-1]
            ra = r[w > 0]
            rC = contact_matrix(ra)
            Ia = x[w > 0]
            infected_roommates = (rC * Ia).sum(1)
            room_count[T-1] = (infected_roommates > 1).sum()

            total_count = np.nansum(X, axis=0)

            floor_counts = []
            for i in range(1,6):
                # does this work with matrix indexing?
                floor_count = np.nansum(X * (self.F == i), axis=0)
                floor_counts.append(floor_count)
            stats = [total_count] + floor_counts + [room_count]
            data = torch.tensor(np.stack(stats)).float() / self.L
            
            # need to tweak this
            if not (self.het or show_full): data = data[0]
            
            return data
        
        else:
            return torch.tensor(X)

        # if self.return_case_count:
        #     case_count = (X == 1).any(1).sum()
        #     return data, case_count


    def sample_logbeta(self, N, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        if self.het:
            mvn = MultivariateNormal(self.prior_mu, self.prior_sigma)
            logbeta = mvn.sample((N,))
        else:
            logbeta = torch.normal(self.prior_mu, self.prior_sigma, (N, 1))
        return logbeta