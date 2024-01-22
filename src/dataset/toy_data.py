import torch
import numpy as np
from torch.utils.data import Dataset
from torch.distributions import MultivariateNormal, Normal

from src.utils import lower_tri

class TestDataset(Dataset):
    # can the density network learn a simple gaussian?
    def __init__(self, n_sample, random_state):
        self.n_sample = n_sample
        self.x, self.theta = self.simulate_data(random_state)
        self.d_x = 5
        self.d_theta = 1

    def __len__(self):
        return self.n_sample

    def __getitem__(self, index):
        return self.x[index], self.theta[index]
    
    def simulate_data(self, random_state):
        torch.manual_seed(random_state)
        theta = torch.normal(1, 1, (self.n_sample,))
        x = torch.normal(0, 3, (self.n_sample, 5))
        return x, theta
    
    def get_observed_data(self):
        torch.manual_seed(4)
        return torch.normal(0,1, size=(5,))
    
    def evaluate(self, mu, sigma, observed_data):
        mu_mse = ((mu - 1) ** 2).mean().item()
        sigma_mse = ((sigma - 1)**2).mean().item()
        print(f"mu, MSE: {mu_mse:.3f}")
        print(f"sigma, MSE: {sigma_mse:.3f}")



class NormalNormalDataset(Dataset):
    def __init__(self, n_obs, n_sample, random_state, shrinkage, noise, dimension):
        if dimension > 3:
            raise ValueError("Test case designed for dimensions < 4")
        self.n_obs = n_obs # something small, like 10 i think
        self.n_sample = n_sample
        self.shrinkage = shrinkage # l2 penalty--smaller shrinks more
        self.noise = noise
        self.d_theta = dimension
        self.d_x = dimension
        THETA_TRUE = torch.tensor([1.45, 1.79, 0.49])
        self.theta_true = THETA_TRUE[:dimension]

        self.x, self.theta = self.simulate_data(random_state)

    def __len__(self):
        return self.n_sample

    def __getitem__(self, index):
        return self.x[index], self.theta[index]
    

    def simulate_data(self, random_state):
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        p = self.d_theta
        mu_0 = torch.zeros(p)
        Sigma_0 = self.shrinkage * torch.eye(p)
        mus =  torch.tensor(
            np.random.multivariate_normal(mu_0, Sigma_0, size=self.n_sample)
        ).float()
        if p == 1:
            mus = mus[:, 0]
            likelihood = Normal(
                mus,
                torch.tensor(self.noise).repeat_interleave(self.n_sample)
                )
            xs = likelihood.rsample((self.n_obs,)).sum(0)
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
                self.theta_true, self.noise, (self.n_obs,)
            ).sum()
        else:
            ll_true = MultivariateNormal(self.theta_true, self.noise *torch.eye(p))
            x_o = ll_true.rsample((self.n_obs,)).sum(0)
        return x_o.unsqueeze(0)
    
    def evaluate(self, mu, sigma, x_o):
        exact_mu, exact_sigma = self.posterior_mean(x_o)
        mu_error = (mu - exact_mu)**2
        sigma_error = (sigma - exact_sigma)**2
        if self.d_theta == 1:
            print(f"mu error: {mu_error.item():.3f}")
            print(f"sigma error: {sigma_error.item():.3f}")
        else:
            # TODO: convert "sigma" from lower tri to cov matrix
            for i in range(self.d_theta):
                print(f"Component {i+1} mu error: {mu_error[0, i].item():.3f}")
            L = lower_tri(sigma, self.d_theta)
            cov_est = L @ L.T
            cov_true = exact_sigma * torch.eye(self.d_theta)
            cov_error = torch.linalg.matrix_norm(cov_est - cov_true)
            print(f"Cov. matrix error: {cov_error.item():.3f}")

    def posterior_mean(self, x_o):
        # sanity check this for MVN
        s02 = self.shrinkage
        s2 = self.noise
        n = self.n_obs
        sigma = 1 / ((1 / s02) + (n / s2))
        mu = sigma * (x_o/s2)
        return mu, sigma



class BayesLinRegDataset(Dataset):
    def __init__(self, n_obs, n_sample, random_state, shrinkage, noise):
        self.n_obs = n_obs # something small, like 10 i think
        self.n_sample = n_sample
        self.shrinkage = shrinkage # l2 penalty--smaller shrinks more
        self.noise = noise
        # may want to generalize this to a higher dim regression
        self.d_theta = 2
        self.d_x = n_obs
        self.theta_true = np.array([1.6, -.9])

        self.data, self.theta = self.simulate_data(random_state)

    def __len__(self):
        return self.n_sample

    def __getitem__(self, index):
        return self.data[index], self.theta[index]
    
    def simulate_data(self, random_state):
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        p = self.d_theta
        m_0 = torch.zeros(p)
        S_0 = self.shrinkage * torch.eye(3)
        betas = np.random.multivariate_normal(m_0, S_0, size=self.n_sample)
        X = np.random.uniform(low = -5, high=5, size=(self.n_obs, p))
        ys = torch.normal(torch.tensor(betas @ X.T), self.noise)
        # TODO: combine ys and xs
        return ys.float(), torch.tensor(betas).float()
    
    def evaluate(self, mu, sigma, y_o):
        raise NotImplementedError
        # mu_mse = (mu ** 2).mean().item()
        # sigma_mse = ((sigma - 1)**2).mean().item()
        # print(f"mu, MSE: {mu_mse:.3f}")
        # print(f"sigma, MSE: {sigma_mse:.3f}")
    
    def posterior_mean(self, xy_0):
        # how to disaggregate x and y?
        pass
    
    def get_observed_data(self):
        np.random.seed(4)
        torch.manual_seed(4)
        X = np.random.uniform(low = -5, high=5, size=(self.n_obs, self.d_theta))
        y_o = torch.normal(
            torch.tensor((self.theta_true @ X.T)), self.noise
        )
        return y_o
    
        


