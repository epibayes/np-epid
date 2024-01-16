import lightning as L
import torch
from torch.nn import Linear
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal


class GaussianDensityNetwork(L.LightningModule):
    def __init__(self, d_x, d_theta, d_model, lr):
        super().__init__()

        # compute number of outputs
        self.n_mu = d_theta
        self.n_sigma = d_theta + d_theta * (d_theta - 1) // 2
        n_outputs = self.n_mu + self.n_sigma

        self.ff = torch.nn.Sequential(
            Linear(d_x, d_model),
            # dropout?
            torch.nn.ReLU(),
            Linear(d_model, d_model),
            torch.nn.ReLU(),
            Linear(d_model, n_outputs),
        )
        # eventually need to save this as an hparam if i am checkpointing models
        self.lr = lr

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        y = self.ff(x)
        mu = y[:, :self.n_mu]
        sigma = torch.exp(y[:, self.n_mu:])
        return mu, sigma
    
    def training_step(self, batch, batch_idx):
        # consider renaming theta to z
        x, theta = batch
        mu, sigma = self(x)
        loss = self.gaussiannll(theta, mu, sigma)
        self.log("train_loss", loss)
        return loss

    
    def validation_step(self, batch, batch_idx):
        x, theta = batch
        mu, sigma = self(x)
        loss = self.gaussiannll(theta, mu, sigma)
        self.log("val_loss", loss)
        return loss

    def gaussiannll(self, theta, mu, sigma):
        p = self.n_mu
        if p == 1:
            normal = Normal(mu, sigma)
            l = - normal.log_prob(theta)
        else:
            b = theta.shape[0]
            L = torch.zeros(b, p, p)
            tril_ix = torch.tril_indices(p, p)
            L[:, tril_ix[0], tril_ix[1]] = sigma
            mvn = MultivariateNormal(loc=mu, scale_tril=L)
            l = - mvn.log_prob(theta)

        return l.mean()
    

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    

    def predict_step(self, x):
        mu, sigma = self(x)
        return mu, sigma
        


# TODO: multiple components (mixture density network)