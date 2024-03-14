import torch
import numpy as np
from .simulator import Simulator



class SIModel(Simulator):
    def __init__(self, alpha, gamma, beta_true, 
                 prior_mu, prior_sigma, n_zones, N, T, 
                 random_state=None, n_sample=None):
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
        if np.isscalar(beta_true):
            beta_true = [beta_true]
        self.beta_true = beta_true
        self.n_sample = n_sample
        self.random_state = random_state
        if n_sample is not None:
            self.data, self.theta = self.simulate_data()


    def simulate_data(self):
        logbetas = self.sample_logbeta(self.n_sample)
        xs = torch.empty((self.n_sample, self.d_x))
        # consider vectorizing if this ends up being slow
        for i in range(self.n_sample):
            xs[i] = self.SI_simulator(logbetas[i], self.random_state)

        return xs, logbetas.float()
    
    def get_observed_data(self):
        # TODO: need to handle case where beta_true is a list
        logbeta_true = torch.log(torch.tensor(self.beta_true))
        x_o = self.SI_simulator(logbeta_true, 29)
        return x_o.unsqueeze(0).float()

    def SI_simulator(self, logbeta, seed=None):
        # beta is infection rate
        assert type(logbeta) is torch.Tensor
        if len(logbeta) == 1:
            logbeta = torch.tensor((logbeta.item(), 0))
        assert len(logbeta) == self.n_zones + 1
        beta = torch.exp(logbeta)
        if seed is not None:
            np.random.seed(seed)
        A  = np.empty((self.N, self.T))
        # assign zones at random
        Z = np.random.choice(np.arange(self.n_zones), self.N)
        # seed initial infections
        A[:, 0] = np.random.binomial(1, self.alpha, self.N)

        for t in range(1, self.T):
            I = A[:, t-1]
            # components dependent on individual covariates
            hazard = I.sum() * beta[0] * np.ones(self.N)
            if self.n_zones > 1:
                Zx, Zy = np.meshgrid(Z, Z)
                # generate contact matrix: each row i indicates who shares a zone with patient i
                C = (Zx == Zy).astype(int)
                hazard += (C * I).sum(1) * beta[Z+1]
                # TODO: introduce room-level risk
            p = 1 - np.exp(-hazard / self.N)
            A[:,t] = np.where(I, np.ones(self.N), np.random.binomial(1, p, self.N))
            discharge = np.random.binomial(1, self.gamma, size=self.N)
            A[:,t] = np.where(discharge, np.random.binomial(1, self.alpha, self.N), A[:, t])

        A = torch.tensor(A).float() # make it all float for good measure
        if self.n_zones == 1:
            return A.mean(0) # proportion of infecteds at each time step
        else:
            zone_counts = [A[Z == i].mean(0) for i in range(self.n_zones)]
            return torch.cat([A.mean(0)] + zone_counts)
    
    def sample_logbeta(self, N):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        if np.isscalar(self.prior_mu) and self.n_zones > 1:
            #TODO: implement this!
            raise NotImplementedError
        else:
            log_beta = torch.normal(self.prior_mu, self.prior_sigma, (N, 1))
        return log_beta

