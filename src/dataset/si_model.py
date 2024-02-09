import torch
import numpy as np

from .simulator import Simulator



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
        self.random_state = random_state
        self.data, self.theta = self.simulate_data()


    def simulate_data(self):
        torch.manual_seed(self.random_state)
        betas = self.sample_beta(self.n_sample)
        xs = torch.empty((self.n_sample, self.d_x))
        # consider vectorizing if this ends up being slow
        for i in range(self.n_sample):
            xs[i] = self.SI_simulator(betas[i], self.random_state)

        return xs, betas.float()
    
    def get_observed_data(self):
        return self.SI_simulator(torch.tensor((self.beta_true,)), self.random_state)

    def SI_simulator(self, beta, seed=None):
        # beta is infection rate
        assert type(beta) is torch.Tensor
        if len(beta) == 1:
            beta = torch.tensor((beta.item(), 0))
        assert len(beta) == self.n_zones + 1
    
        if seed is not None:
            torch.manual_seed(seed)
        A  = torch.empty((self.N, self.T)).int()
        # assign zones at random
        Z = torch.multinomial(torch.ones(self.n_zones), num_samples=self.N, replacement=True)
        # TODO: introduce random room assignments by floor
        # seed initial infections
        A[:, 0] = torch.bernoulli(torch.full((self.N,), self.alpha))
        for t in range(1, self.T):
            I = A[:, t-1]
            # components dependent on individual covariates
            hazard = I.sum() * torch.full((self.N,), beta[0])
            if self.n_zones > 1:
                raise NotImplementedError
                Zx, Zy = np.meshgrid(Z, Z)
                # generate contact matrix: each row i indicates who shares a zone with patient i
                C = (Zx == Zy).astype(int)
                hazard += (C * I).sum(1) * beta[Z+1]
            # discharge = np.random.binomial(1, self.gamma, size=self.N)
            discharge = torch.bernoulli(torch.full((self.N,), self.gamma)).int()
            p = 1 - torch.exp(-hazard / self.N) # infection probability
            # if a susceptible person is not discharged, they may experience an infection event
            A[:, t] = torch.where(((1 - I) * (1 - discharge)).bool(), torch.bernoulli(p).int(), A[:, t])
            # if a person is discharged, replace them with an infected or susceptible
            A[:,t] = torch.where(discharge.bool(), torch.bernoulli(torch.full((self.N,), self.alpha)).int(), A[:, t])

        A = A.float()
        if self.n_zones == 1:
            return A.mean(0) # proportion of infecteds at each time step
        else:
            zone_counts = [A[Z == i].mean(0) for i in range(self.n_zones)]
            return torch.tensor([A.mean(0)] + zone_counts)
    
    def sample_beta(self, N):
        if np.isscalar(self.prior_mu) and self.n_zones > 1:
            raise NotImplementedError
            # log_beta = np.random.normal(self.prior_mu, self.prior_sigma, self.n_zones + 1)
        else:
            log_beta = torch.normal(self.prior_mu, self.prior_sigma, (N, 1))
        return torch.exp(log_beta)

