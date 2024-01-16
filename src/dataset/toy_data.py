import torch
import numpy as np
from torch.utils.data import Dataset
from torch.distributions import MultivariateNormal, Normal

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
    
    def evaluate(self, mu, sigma):
        mu_mse = ((mu - 1) ** 2).mean().item()
        sigma_mse = ((sigma - 1)**2).mean().item()
        print(f"mu, MSE: {mu_mse:.3f}")
        print(f"sigma, MSE: {sigma_mse:.3f}")



class NormalNormalDataset(Dataset):
    def __init__(self, n_obs, n_sample, random_state, shrinkage, noise, dimension):
        self.n_obs = n_obs # something small, like 10 i think
        self.n_sample = n_sample
        self.shrinkage = shrinkage # l2 penalty--smaller shrinks more
        self.noise = noise
        self.d_theta = dimension
        self.d_x = n_obs
        self.theta_true = 1.64 # p-vector of 1.64 in all dimension

        self.x, self.theta = self.simulate_data(random_state)

    def __len__(self):
        return self.n_sample

    def __getitem__(self, index):
        return self.x[index], self.theta[index]
    

    def simulate_data(self, random_state):
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        # TODO: can I do this in pure torch?
        p = self.d_theta
        mu_0 = torch.zeros(p)
        Sigma_0 = self.shrinkage * torch.diag(torch.ones(p))
        mus =  np.random.multivariate_normal(mu_0, Sigma_0, size=self.n_sample)
        if p == 1:
            mus = torch.tensor(mus[:, 0])
            likelihood = Normal(
                mus,
                torch.tensor(self.noise).repeat_interleave(self.n_sample)
                )
            xs = likelihood.rsample((self.n_obs,)).transpose(0,1)
        else:
            raise NotImplementedError
            # cov = self.sigma * torch.diag(torch.ones(p))
            # cov = cov_matrix.unsqueeze(0).repeat(self.n_sample, 1, 1)
            # likelihood = MultivariateNormal(mus, cov_matrix)
        return xs.float(), mus.float()
    
    def get_observed_data(self):
        torch.manual_seed(4)
        # make sure this is MVN
        x_o = torch.normal(
            self.theta_true, self.noise, (self.n_obs,)
        )
        return x_o
    
    def evaluate(self, mu, sigma):
        x_o = self.get_observed_data()
        exact_mu, exact_sigma = self.posterior_mean(x_o)
        mu_error = (mu - exact_mu)**2
        sigma_error = (sigma - exact_sigma)**2
        print(f"mu error: {mu_error.item():.3f}")
        print(f"sigma error: {sigma_error.item():.3f}")

    def posterior_mean(self, x_o):
        s02 = self.shrinkage
        s2 = self.noise
        n = x_o.shape[0]
        sigma = 1 / ((1 / s02) + (n / s2))
        mu = sigma * (x_o.sum()/s2)
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
        Sigma_0 = self.shrinkage * torch.diag(torch.ones(p))
        betas = np.random.multivariate_normal(mu_0, Sigma_0, size=self.n_sample)
        X = np.random.uniform(low = -5, high=5, size=(self.n_obs, p))
        ys = torch.normal(torch.tensor(betas @ X.T), self.noise)
        return ys.float(), betas
    
    def evaluate(self, mu, sigma):
        posterior_mean = mu.mean().item()
        # mu_mse = (mu ** 2).mean().item()
        # sigma_mse = ((sigma - 1)**2).mean().item()
        # print(f"mu, MSE: {mu_mse:.3f}")
        # print(f"sigma, MSE: {sigma_mse:.3f}")
    
    def get_observed_data(self):
        np.random.seed(4)
        torch.manual_seed(4)
        X = np.random.uniform(low = -5, high=5, size=(32, self.n_obs, self.d_theta))
        y_0 = torch.normal(
            torch.tensor((self.theta_true @ np.transpose(X, (0, 2, 1)))), self.noise
        )
        return y_0.float()
    
        


