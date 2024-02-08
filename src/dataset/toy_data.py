import torch
import numpy as np
from torch.distributions import MultivariateNormal, Normal

from src.utils import lower_tri, Simulator

COEFFS = torch.tensor([1.45, 1.79, 0.49])

class TestDataset(Simulator):
    # can the density network learn a simple gaussian?
    def __init__(self, n_sample, random_state):
        self.n_sample = n_sample
        self.d_x = 5
        self.d_theta = 1
        self.data, self.theta = self.simulate_data(random_state)

    
    def simulate_data(self, random_state):
        torch.manual_seed(random_state)
        theta = torch.normal(1, 1, (self.n_sample,))
        x = torch.normal(0, 3, (self.n_sample, 5))
        return x, theta.unsqueeze(1)
    
    def get_observed_data(self):
        torch.manual_seed(4)
        return torch.normal(0,1, size=(1, 5))
    
    def evaluate(self, mu, sigma, observed_data):
        mu_mse = ((mu - 1) ** 2).mean().item()
        sigma_mse = ((sigma - 1)**2).mean().item()
        print(f"mu, MSE: {mu_mse:.3f}")
        print(f"sigma, MSE: {sigma_mse:.3f}")



class NormalNormalDataset(Simulator):
    def __init__(self, n_obs, n_sample, random_state, shrinkage, noise, dimension):
        if dimension > 3:
            raise ValueError("Test case designed for dimensions < 4")
        self.n_sample = n_sample
        self.n_obs = n_obs # something small, like 10 i think
        self.shrinkage = shrinkage # l2 penalty: *std* of prior
        self.noise = noise # *std* of observational error
        self.d_theta = dimension
        self.d_x = dimension
        self.theta_true = COEFFS[:dimension]

        self.data, self.theta = self.simulate_data(random_state)

    

    def simulate_data(self, random_state):
        np.random.seed(random_state)
        torch.manual_seed(random_state)
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
        torch.manual_seed(4)
        p = self.d_theta
        if p == 1:
            x_o = torch.normal(
                self.theta_true.item(), self.noise, (self.n_obs,)
            ).sum().unsqueeze(0)
        else:
            ll_true = MultivariateNormal(self.theta_true, self.noise *torch.eye(p))
            x_o = ll_true.rsample((self.n_obs,)).sum(0)
        return x_o.unsqueeze(0)
    
    def evaluate(self, mu, sigma, x_o):
        exact_mu, exact_sigma = self.posterior_mean(x_o)
        if self.d_theta == 1:
            print(f"Approximate posterior: B ~ N({mu.item():.3f}, {sigma.item():.3f})")
            print(f"Exact posterior: B ~ N({exact_mu.item():.3f}, {exact_sigma:.3f})")
        else:
            mu_error = (mu - exact_mu)**2
            print(f"Average mu error: {mu_error.mean().item():.3f}")
            L = lower_tri(sigma, self.d_theta)
            cov_est = L @ L.T
            cov_true = exact_sigma * torch.eye(self.d_theta)
            cov_error = torch.linalg.matrix_norm(cov_est - cov_true)
            print(f"Cov. matrix error: {cov_error.item():.3f}")

    def posterior_mean(self, x_o):
        # sanity check this for MVN
        s02 = self.shrinkage ** 2
        s2 = self.noise
        n = self.n_obs
        sigma = 1 / ((1 / s02) + (n / s2))
        mu = sigma * (x_o/s2)
        return mu, sigma



class BayesLinRegDataset(Simulator):
    def __init__(self, n_obs, n_sample, random_state, shrinkage, noise, dimension):
        if dimension > 3:
            raise ValueError("Test case designed for dimensions < 4")
        self.n_sample = n_sample
        self.n_obs = n_obs # something small, like 10 i think
        self.shrinkage = shrinkage # l2 penalty--smaller shrinks more
        self.noise = noise
        self.d_theta = dimension
        self.d_x = dimension * (dimension + 1)
        self.theta_true = COEFFS[:dimension]

        self.data, self.theta = self.simulate_data(random_state)

    def simulate_data(self, random_state):
        np.random.seed(random_state)
        torch.manual_seed(random_state)
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
    
    def evaluate(self, mu, sigma, data_o):
        data_o = data_o[0]
        xtx_o = data_o[:-self.d_theta]
        xty_o = data_o[-self.d_theta:]
        exact_mu, exact_sigma = self.posterior_mean(xtx_o, xty_o)
        if self.d_theta == 1:
            print(f"Approximate posterior: B ~ N({mu.item():.3f}, {sigma.item():.3f})")
            print(f"Exact posterior: B ~ N({exact_mu.item():.3f}, {exact_sigma.item():.3f})")
        else:
            pass
        # TODO: figure out how to evaluate multidim case
        
        
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
        np.random.seed(4)
        torch.manual_seed(4)
        X = np.random.uniform(low = -5, high=5, size=(self.n_obs, self.d_theta))
        y_o = torch.normal(
            torch.tensor((self.theta_true @ X.T)).clone().detach(),
              self.noise
        )
        xtx = torch.tensor((X.T @ X).flatten())
        xty = (y_o @ X)
        return torch.cat((xtx, xty)).unsqueeze(0).float()
    
        


