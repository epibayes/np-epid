import torch
import numpy as np
from .simulator import Simulator
from torch.distributions import MultivariateNormal
from ..utils import contact_matrix

class SIModel(Simulator):
    def __init__(self, alpha, gamma, beta_true, heterogeneous,
                 prior_mu, prior_sigma,  N, T, summarize=False,
                 observed_seed=None, room=False, n_sample=None,
                 flatten=True):
        self.alpha = alpha # baseline proportion infected in pop
        self.gamma = gamma # discharge rate
        if np.isscalar(beta_true):
            self.beta_true = [beta_true]
        else:
            self.beta_true = beta_true
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
            rs = 5 * i
            xs[i] = self.SI_simulator(
                np.array(logbetas[i]), rs).flatten()

        return xs, logbetas.float()
    
    def get_observed_data(self, observed_seed=None):
        if observed_seed is None:
            observed_seed = self.obs
        logbeta_true = torch.log(torch.tensor(self.beta_true))
        x_o = self.SI_simulator(
            np.array(logbeta_true), observed_seed)
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
        if seed is not None:
            np.random.seed(seed)
        A  = np.empty((self.N, self.T))
        # assign zones at random
        F = np.arange(self.N) % 5 # does this need to be random?
        R = np.arange(self.N) % (self.N // 2) # N should be divisible by 10
        # seed initial infections
        A[:, 0] = np.random.binomial(1, self.alpha, self.N)
        room_infect_density = np.ones(self.T)
        fC = contact_matrix(F)
        rC = contact_matrix(R)
        for t in range(1, self.T):
            I = A[:, t-1]
            # components dependent on individual covariates
            hazard = I.sum() * beta[0] * np.ones(self.N)
            if self.het:
                hazard += (fC * I).sum(1) * beta[F+1]
                infected_roommates = (rC * I).sum(1)
                # is there a better summary statistic?
                # TODO: this is a major bug!
                if infected_roommates.max() == 0:
                    room_infect_density[t] = 1
                else:
                    room_infect_density[t] = infected_roommates[infected_roommates > 0].mean()
                hazard += infected_roommates * beta[-1]
            p = 1 - np.exp(-hazard / self.N)
            # if someone is not yet infected, simulate transmission event
            A[:,t] = np.where(I, np.ones(self.N), np.random.binomial(1, p, self.N))
            discharge = np.random.binomial(1, self.gamma, size=self.N)
            A[:,t] = np.where(discharge, np.random.binomial(1, self.alpha, self.N), A[:, t])

        A = torch.tensor(A).float() # make it all float for good measure
        # is it as simple as offsetting by first element of A?
        w = None if self.summarize else 0
        total_count = A.mean(w)
        if self.het:
            floor_counts = [A[F == i].mean(w) for i in range(5)]
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
        if self.d_theta == 1:
            log_beta = torch.normal(self.prior_mu, self.prior_sigma, (N, 1))
        else:
            mvn = MultivariateNormal(self.prior_mu, self.prior_sigma)
            log_beta = mvn.sample((N,))
        return log_beta

