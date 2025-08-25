import torch
import numpy as np
from numpy.random import binomial
from .simulator import Simulator
from torch.distributions import MultivariateNormal
from ..utils import contact_matrix

class SIModel(Simulator):
    def __init__(self, alpha, gamma, beta_true, heterogeneous,
                 prior_mu, prior_sigma,  N, T, name=None, summarize=False,
                 observed_seed=None, room=False, log_scale=True,
                 n_sample=None, flatten=True, pi=None, eta = None):
        self.log_scale = log_scale
        self.alpha = alpha # baseline proportion infected in pop
        self.gamma = gamma # discharge rate
        if np.isscalar(beta_true):
            self.beta_true = [beta_true]
        else:
            self.beta_true = beta_true
        if pi is not None:
            if np.isscalar(pi):
                self.pi = np.array([pi])
            else:
                self.pi = np.array(pi)
            assert len(self.beta_true) == len(self.pi)
        else:
            self.pi = pi
        if eta is None:
            self.eta = 1
        else:
            self.eta = eta
        self.N = N
        self.T = T
        self.het = heterogeneous
        self.room = room
        self.cap = np.array([N] + [60 for _ in range(5)] + [2])
        self.d_theta = 7 if self.het else 1
        assert self.d_theta == len(self.beta_true)
        self.set_prior(prior_mu, prior_sigma)
        self.flatten = flatten
        if summarize: 
            self.d_x = T * self.d_theta # gets flattened
        else:
            self.d_x = T # trailing dimension of a NxT matrix
        self.summarize = summarize
        self.n_sample = n_sample
        self.F = np.repeat(
            (np.arange(self.N) % 5)[:, np.newaxis], T, 1
        )
        self.R = np.repeat(
            (np.arange(self.N) % (self.N // 2))[:, np.newaxis], T, 1
        )
            
        if n_sample is not None:
            self.data, self.theta = self.simulate_data()
        self.obs = observed_seed
        self.name = name

    def set_prior(self, mu, sigma):
        if self.d_theta == 1:
            self.prior_mu = mu
            self.prior_sigma = sigma
        else:
            self.prior_mu = torch.full((7,), mu).float()
            self.prior_sigma = torch.diag(torch.full((7,), sigma)).float()

    def simulate_data(self):
        thetas = self.sample_logbeta(self.n_sample, seed=5)
        if not self.log_scale:
            thetas = np.exp(thetas)
        if self.summarize:
            xs = torch.empty((self.n_sample, self.d_x))
        else:
            xs = torch.empty(self.n_sample, self.N, self.T)
        for i in range(self.n_sample):
            random_seed = 5 * i
            sim = self.SI_simulator(
                np.array(thetas[i]),
                random_seed, self.summarize
            )
            xs[i] = sim.flatten() if self.summarize else sim 

        return xs, thetas.float()
    
    def get_observed_data(self, observed_seed=None):
        # why did I include observed_seed as an argument?
        if observed_seed is None:
            observed_seed = self.obs
        theta_true = np.array(self.beta_true)
        if self.log_scale:
            theta_true = np.log(theta_true)
        x_o = self.SI_simulator(theta_true, observed_seed)
        if not self.het:
            x_o = x_o.unsqueeze(0)
        if self.flatten and self.het:
            x_o = x_o.unsqueeze(0)
        return x_o.float()

    def SI_simulator(self, theta, seed=None, summarize=False):
        if self.log_scale:
            beta = np.exp(theta)
        else:
            beta = theta
        if len(beta) == 1:
            beta = np.array(((beta[0],)))
        assert len(beta) == self.d_theta
        if self.pi is not None:
            beta = beta * self.pi
        if seed is not None:
            np.random.seed(seed)
        X  = np.empty((self.N, self.T))
        Y = np.empty((self.N, self.T))
        # assign zones
        F = self.F.T[0] # np.arange(self.N) % 5
        R = self.R.T[0] # np.arange(self.N) % (self.N // 2) # N should be divisible by 10
        # seed initial infections
        X[:, 0] = np.random.binomial(1, self.alpha, self.N)
        Y[:, 0] = X[:, 0]
        room_count = np.ones(self.T)
        fC = contact_matrix(F)
        rC = contact_matrix(R)
        for t in range(1, self.T):
            I = X[:, t-1]
            hazard = I.sum() * beta[0] * np.ones(self.N) / self.cap[0]
            if self.het:
                hazard += (fC * I).sum(1) * beta[F+1] / self.cap[F+1]
                infected_roommates = (rC * I).sum(1)
                hazard += infected_roommates * beta[-1] / self.cap[-1]
                roommates_obs = (rC * Y[:, t-1]).sum(1) 
                assert (roommates_obs <= infected_roommates).all()
                room_count[t-1] = (roommates_obs > 1).sum()
            p = 1 - np.exp(-hazard)
            new_infections = np.random.binomial(1, p, self.N)
            X[:, t] = np.where(I, np.ones(self.N), new_infections)
            if self.eta == 1:
                observed = np.ones(self.N)
            else:
                observed = binomial(1, self.eta, self.N)
            Y[:, t] = np.where(
                X[:, t] * (1 - Y[:, t-1]),
                observed,
                Y[:, t-1]
            )
            if self.eta == 1:
                assert (X[:, t] == Y[:, t]).all()
            # if someone is not yet infected, simulate transmission event
            discharge = np.random.binomial(1, self.gamma, size=self.N)
            screening = np.random.binomial(1, self.alpha, self.N)
            X[:, t] = np.where(discharge, screening, X[:, t])
            Y[:, t] = np.where(discharge, screening, Y[:, t])
            
        if self.het:
            roommates_obs = (rC * Y[:, -1]).sum(1)
            room_count[-1] = (roommates_obs > 1).sum()

        Y = torch.tensor(Y).float() # make it all float for good measure
        # is it as simple as offsetting by first element of A?
        if self.summarize:
            total_count = Y.mean(0)
            floor_counts = [Y[F == i].mean(0) for i in range(5)]
            room_count = room_count / self.N
            if self.het:
                data = torch.stack([total_count] + floor_counts + [torch.tensor(room_count)])
            else:
                data = total_count
            if self.flatten:
                return data.flatten()
            else:
                return data
        else:
            return Y

    
    def sample_logbeta(self, N, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        if self.het:
            mvn = MultivariateNormal(self.prior_mu, self.prior_sigma)
            log_beta = mvn.sample((N,))
        else:
            log_beta = torch.normal(self.prior_mu, self.prior_sigma, (N, 1))
        return log_beta