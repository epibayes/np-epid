import torch
import numpy as np

from src.utils import Simulator



class SIModel(Simulator):
    def __init__(self, n_sample, random_state, alpha, gamma, beta_true, 
                 prior_mu, prior_sigma, n_zones, N, T):
        self.alpha = alpha # baseline proportion infected in pop
        self.gamma = gamma # discharge rate
        self.N = N
        self.T = T
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        # observed data
        self.n_zones = n_zones
        self.d_theta = 1 if n_zones == 1 else 1 + n_zones
        self.d_x = T * self.d_theta
        # self.x_o = self.simulate_SIS(beta_true, seed=29)
        self.beta_true = beta_true
        self.n_sample = n_sample
        self.data, self.theta = self.simulate_data(random_state)


    def simulate_data(self, random_state):
        torch.manual_seed(random_state)
        betas = self.sample_beta(self.n_sample)
        xs = torch.empty((self.n_sample, self.d_x))
        # consider vectorizing if this ends up being slow
        for i in range(self.n_sample):
            xs[i] = self.SI_simulator(betas[i], random_state)

        return xs, betas.float()

    def SI_simulator(self, beta, seed=None):
        # beta is infection rate
        if type(beta) is float:
            beta = np.array((beta, 0))
        else:
            beta = np.array(beta)

        assert len(beta) == self.n_zones + 1
    
        if seed is not None:
            np.random.seed(seed)
        A  = np.empty((self.N, self.T + 1))
        # assign zones at random
        Z = np.random.choice(np.arange(self.n_zones), self.N)
        # seed initial infections
        A[:, 0] = np.random.binomial(1, self.alpha, self.N)

        for t in range(1, self.T+1):
            I = A[:, t-1]
            # components dependent on individual covariates
            hazard = I.sum() * beta[0] * np.ones(self.N)
            if self.n_zones > 1:
                raise NotImplementedError
                Zx, Zy = np.meshgrid(Z, Z)
                # generate contact matrix: each row i indicates who shares a zone with patient i
                C = (Zx == Zy).astype(int)
                hazard += (C * I).sum(1) * beta[Z+1]
                # TODO: introduce room-level risk
            p = 1 - np.exp(-hazard / self.N)
            A[:,t] = np.where(I, np.ones(self.N), np.random.binomial(1, p, self.N))
            discharge = np.random.binomial(1, self.gamma, size=self.N)
            A[:,t] = np.where(discharge, np.random.binomial(1, self.alpha, self.N), A[:, t])

        if self.n_zones == 1:
            return A.mean(0) # proportion of infecteds at each time step
        else:
            zone_counts = [A[Z == i].mean(0) for i in range(self.n_zones)]
            return np.array([A.mean(0)] + zone_counts)
    
    def sample_beta(self, N):
        if np.isscalar(self.prior_mu) and self.n_zones > 1:
            raise NotImplementedError
            # log_beta = np.random.normal(self.prior_mu, self.prior_sigma, self.n_zones + 1)
        else:
            log_beta = torch.normal(self.prior_mu, self.prior_sigma, (N, 1))
        return torch.exp(log_beta)

