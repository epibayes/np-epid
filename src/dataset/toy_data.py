import torch
import numpy as np
from torch.distributions import MultivariateNormal, Normal
import math
from .simulator import Simulator

COEFFS = torch.tensor([1.45, 1.79, 0.49])


class NormalNormalDataset(Simulator):
    def __init__(self, n_obs, n_sample, observed_seed, shrinkage, noise, dimension, name):
        if dimension > 3:
            raise ValueError("Test case designed for dimensions < 4")
        self.n_sample = n_sample
        self.observed_seed = observed_seed
        self.n_obs = n_obs # something small, like 10 i think
        self.shrinkage = shrinkage # l2 penalty: *std* of prior
        self.noise = noise # *std* of observational error
        self.d_theta = dimension
        self.d_x = dimension
        self.theta_true = COEFFS[:dimension]

        self.data, self.theta = self.simulate_data()
        self.name = name

    

    def simulate_data(self):
        np.random.seed(10)
        torch.manual_seed(10)
        p = self.d_theta
        mu_0 = torch.zeros(p)
        Sigma_0 = self.shrinkage**2 * torch.eye(p)
        mus =  torch.tensor(
            np.random.multivariate_normal(mu_0, Sigma_0, size=self.n_sample)
        ).float()
        if p == 1:
            mus = mus[:, 0]
            likelihood = Normal(
                mus,
                torch.tensor(self.noise).repeat_interleave(self.n_sample)
                )
            xs = likelihood.rsample((self.n_obs,)).sum(0).unsqueeze(1)
            mus = mus.unsqueeze(1)
        else:
            cov = self.noise * torch.eye(p)
            likelihood = MultivariateNormal(mus, cov)
            xs = likelihood.rsample((self.n_obs,)).sum(0)
        return xs.float(), mus.float()
    
    def get_observed_data(self):
        torch.manual_seed(self.observed_seed)
        p = self.d_theta
        if p == 1:
            x_o = torch.normal(
                self.theta_true.item(), self.noise, (self.n_obs,)
            ).sum().unsqueeze(0)
        else:
            ll_true = MultivariateNormal(self.theta_true, self.noise *torch.eye(p))
            x_o = ll_true.rsample((self.n_obs,)).sum(0)
        return x_o.unsqueeze(0)
    
    def evaluate(self, posterior_params):
        mu, sigma = posterior_params
        x_o = self.get_observed_data()
        exact_mu, exact_sigma = self.posterior_mean(x_o)
        if self.d_theta == 1:
            print(f"Approximate posterior: B ~ N({mu.item():.3f}, {sigma.item():.3f})")
            print(f"Exact posterior: B ~ N({exact_mu.item():.3f}, {exact_sigma:.3f})")
        else:
            mu_error = (mu - exact_mu)**2
            print(f"Average mu error: {mu_error.mean().item():.3f}")
            sigma_error = (sigma - exact_sigma)**2
            print(f"Average sigma error: {sigma_error.mean().item():.3f}")

    def posterior_mean(self, x_o):
        # sanity check this for MVN
        s02 = self.shrinkage ** 2
        s2 = self.noise
        n = self.n_obs
        sigma = 1 / ((1 / s02) + (n / s2))
        mu = sigma * (x_o/s2)
        return mu, sigma



class BayesLinRegDataset(Simulator):
    def __init__(self, n_obs, n_sample, observed_seed, shrinkage, noise, dimension, name):
        if dimension > 3:
            raise ValueError("Test case designed for dimensions < 4")
        self.n_sample = n_sample
        self.observed_seed = observed_seed
        self.n_obs = n_obs # something small, like 10 i think
        self.shrinkage = shrinkage # l2 penalty--smaller shrinks more
        self.noise = noise
        self.d_theta = dimension
        self.d_x = dimension * (dimension + 1)
        self.theta_true = COEFFS[:dimension]

        self.data, self.theta = self.simulate_data()
        self.name = name

    def simulate_data(self):
        # might be nice not to hard code this
        np.random.seed(9)
        torch.manual_seed(9)
        p = self.d_theta
        m_0 = torch.zeros(p)
        S_0 = self.shrinkage**2 * torch.eye(p)
        betas = np.random.multivariate_normal(m_0, S_0, size=self.n_sample)
        X = np.random.uniform(low = -5, high=5, size=(self.n_obs, p))
        ys = torch.normal(torch.tensor(betas @ X.T), self.noise)
        xtx = torch.tensor(np.tile((X.T @ X).flatten(), (self.n_sample, 1)))
        xty = ys @ X # (n_samples, p)
        data = torch.cat((xtx, xty), 1) # (n_samples, p(p+1))
        return data.float(), torch.tensor(betas).float()
    
    def evaluate(self, posterior_params):
        mu, sigma = posterior_params
        data_o = self.get_observed_data()[0]
        xtx_o = data_o[:-self.d_theta]
        xty_o = data_o[-self.d_theta:]
        exact_mu, exact_sigma = self.posterior_mean(xtx_o, xty_o)
        if self.d_theta == 1:
            print(f"Approximate posterior: B ~ N({mu.item():.3f}, {sigma.item():.3f})")
            print(f"Exact posterior: B ~ N({exact_mu.item():.3f}, {exact_sigma.item():.3f})")
        else:
            pass
        
        
    def posterior_mean(self, xtx_o, xty_o):
        p = self.d_theta
        # prior covariance
        S_0 = self.shrinkage**2 * torch.eye(p)
        # likelihood variance
        sigma2 = self.noise**2
        # posterior covariance

        S_N = torch.linalg.inv(xtx_o.unflatten(0, (p, p)) / sigma2 +\
                                torch.linalg.inv(S_0))
        # posterior mean
        m_N = S_N @ (xty_o / sigma2)
        return m_N, S_N
    
    def get_observed_data(self):
        np.random.seed(self.observed_seed)
        torch.manual_seed(self.observed_seed)
        X = np.random.uniform(low = -5, high=5, size=(self.n_obs, self.d_theta))
        y_o = torch.normal(
            torch.tensor((self.theta_true @ X.T)).clone().detach(),
              self.noise
        )
        xtx = torch.tensor((X.T @ X).flatten())
        xty = (y_o @ X)
        return torch.cat((xtx, xty)).unsqueeze(0).float()
    


class ConditionalMoonsDataset(Simulator):
    # test data set for conditional normalizing flows
    def __init__(self, n_sample, name):
        self.n_sample = n_sample
        self.d_x = 2
        self.d_theta = 2
        self.data, self.theta = self.simulate_data()
        self.name = name
         
    def simulate_data(self):
        np.random.seed(8)
        theta = np.random.uniform(-1, 1, (2, self.n_sample))
        a = np.random.uniform(low=-math.pi/2, high=math.pi/2, size=self.n_sample)
        r = np.random.normal(loc=0.1, scale=0.01, size=self.n_sample)
        p = np.stack([r * np.cos(a) + 0.25, r * np.sin(a)])
        b0 = - np.abs(theta[0] + theta[1]) / math.sqrt(2)
        b1 = (-theta[0] + theta[1]) / math.sqrt(2)
        x = np.stack([p[0] + b0, p[1] + b1])
        return torch.tensor(x.T).float(), torch.tensor(theta.T).float()
    
    def get_observed_data(self):
        return torch.tensor([0,0]).unsqueeze(0).float()
        
