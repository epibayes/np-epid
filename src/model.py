import lightning as L
import torch
from torch.nn import Linear, Dropout, ReLU
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

from src.utils import lower_tri


class GaussianDensityNetwork(L.LightningModule):
    def __init__(self, d_x, d_theta, d_model, dropout, lr):
        super().__init__()

        # compute number of outputs
        self.dim = d_theta
        n_sigma = d_theta + d_theta * (d_theta - 1) // 2
        n_outputs = self.dim + n_sigma

        self.ff = torch.nn.Sequential(
            Linear(d_x, d_model),
            Dropout(dropout), ReLU(),
            Linear(d_model, d_model),
            Dropout(dropout), ReLU(),
            Linear(d_model, d_model),
            Dropout(dropout), ReLU(),
            Linear(d_model, n_outputs),
        )
        # eventually need to save this as an hparam if i am checkpointing models
        self.lr = lr

    def forward(self, x):
        assert len(x.shape) == 2
        # assumes shape (n_batches, d_x)
        # d_x could be 
        y = self.ff(x)
        mu = y[:, :self.dim]
        sigma = torch.exp(y[:, self.dim:])
        return mu, sigma
    
    def training_step(self, batch, batch_idx):
        x, theta = batch
        mu, sigma = self(x)
        loss = self.gaussiannll(theta, mu, sigma)
        self.log("train_loss", loss)
        return loss

    
    def validation_step(self, batch, batch_idx):
        # TODO: might be helpful to log how the estimated posterior evolves over time
        # would entail passing in the "observed" data though
        x, theta = batch
        assert len(theta.shape) > 1
        mu, sigma = self(x)
        loss = self.gaussiannll(theta, mu, sigma)
        self.log("val_loss", loss)
        return loss

    def gaussiannll(self, theta, mu, sigma):
        p = self.dim
        if p == 1:
            normal = Normal(mu, sigma)
            l = - normal.log_prob(theta)
        else:
            L = lower_tri(sigma, p)
            mvn = MultivariateNormal(loc=mu, scale_tril=L)
            l = - mvn.log_prob(theta)

        return l.mean()
    

    def configure_optimizers(self):
        # TODO: consider swapping out for SGD with 
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    

    def predict_step(self, x):
        mu, sigma = self(x)
        return mu, sigma