import torch
import numpy as np
from numpy.random import binomial
from .simulator import Simulator
from torch.distributions import MultivariateNormal
from ..utils import contact_matrix

class SIModel(Simulator):
    def __init__(self, alpha, gamma, beta_true, heterogeneous,
                 prior_mu, prior_sigma,  N, T, summarize=False,
                 observed_seed=None, room=False, n_sample=None,
                 flatten=True, pi=None, eta = None):
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
        self.d_theta = 7 if self.het else 1
        assert self.d_theta == len(self.beta_true)
        self.set_prior(prior_mu, prior_sigma)
        self.flatten = flatten
        if summarize: 
            self.d_x = self.d_theta
        else:
            self.d_x = T * self.d_theta
        self.summarize = summarize
        self.n_sample = n_sample
        if n_sample is not None:
            self.data, self.theta = self.simulate_data()
        self.obs = observed_seed

    def set_prior(self, mu, sigma):
        if self.d_theta == 1:
            self.prior_mu = mu
            self.prior_sigma = sigma
            return
        else:
            self.prior_mu = torch.tensor(mu).float()
            self.prior_sigma = torch.diag(torch.tensor(sigma)).float()
            return

    def simulate_data(self):
        logbetas = self.sample_logbeta(self.n_sample, seed=5)
        xs = torch.empty((self.n_sample, self.d_x))
        for i in range(self.n_sample):
            random_seed = 5 * i
            xs[i] = self.SI_simulator(
                np.array(logbetas[i]),
                random_seed,
            ).flatten()

        return xs, logbetas.float()
    
    def get_observed_data(self, observed_seed=None):
        if observed_seed is None:
            observed_seed = self.obs
        logbeta_true = np.log(np.array(self.beta_true))
        x_o = self.SI_simulator(np.array(logbeta_true), observed_seed)
        if self.summarize:
            x_o = x_o.unsqueeze(0)
        if not self.het:
            x_o = x_o.unsqueeze(0)
        if self.flatten and self.het:
            x_o = x_o.unsqueeze(0)
        return x_o.float()

    def SI_simulator(self, logbeta, seed=None):
        beta = np.exp(logbeta)
        if len(beta) == 1:
            beta = np.array(((beta[0],)))
        assert len(beta) == self.d_theta
        if self.pi is not None:
            beta = beta * self.pi
        if seed is not None:
            np.random.seed(seed)
        X  = np.empty((self.N, self.T))
        Y = np.empty((self.N, self.T))
        # assign zones at random
        F = np.arange(self.N) % 5 # does this need to be random?
        R = np.arange(self.N) % (self.N // 2) # N should be divisible by 10
        # seed initial infections
        X[:, 0] = np.random.binomial(1, self.alpha, self.N)
        Y[:, 0] = X[:, 0]
        room_infect_density = np.ones(self.T)
        fC = contact_matrix(F)
        rC = contact_matrix(R)
        for t in range(1, self.T):
            I = X[:, t-1]
            # components dependent on individual covariates
            hazard = I.sum() * beta[0] * np.ones(self.N)
            if self.het:
                hazard += (fC * I).sum(1) * beta[F+1]
                infected_roommates = (rC * I).sum(1)
                hazard += infected_roommates * beta[-1]
                roommates_obs = (rC * Y[:, t-1]).sum(1)
                assert (roommates_obs <= infected_roommates).all()
                if roommates_obs.max() == 0:
                    room_infect_density[t] = 1
                else:
                    room_infect_density[t] = roommates_obs[roommates_obs > 0].mean()
            p = 1 - np.exp(-hazard / self.N)
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
            # if someone is not yet infected, simulate transmission event
            discharge = np.random.binomial(1, self.gamma, size=self.N)
            screening = np.random.binomial(1, self.alpha, self.N)
            X[:, t] = np.where(discharge, screening, X[:, t])
            Y[:, t] = np.where(discharge, screening, Y[:, t])

        Y = torch.tensor(Y).float() # make it all float for good measure
        # is it as simple as offsetting by first element of A?
        w = None if self.summarize else 0
        total_count = Y.mean(w)
        if self.het:
            floor_counts = [Y[F == i].mean(w) for i in range(5)]
            if self.summarize:
                room_infect_density = room_infect_density.mean()
            data =  torch.stack([total_count] + floor_counts + [torch.tensor(room_infect_density)])
        else:
            data = total_count
            
        if self.flatten:
            data = data.flatten()
        
        return data

    
    def sample_logbeta(self, N, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        if self.het:
            mvn = MultivariateNormal(self.prior_mu, self.prior_sigma)
            log_beta = mvn.sample((N,))
        else:
            log_beta = torch.normal(self.prior_mu, self.prior_sigma, (N, 1))
        return log_beta