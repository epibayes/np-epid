import lightning as L
import torch
from torch.nn import Linear, ReLU
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

from src.utils import lower_tri, diag


class GaussianDensityNetwork(L.LightningModule):
    def __init__(self, d_x, d_theta, d_model, lr, weight_decay,
                 mean_field):
        super().__init__()
        # compute number of outputs
        self.dim = d_theta
        # assume diagonal covariance matrix
        if mean_field:
            n_outputs = self.dim * 2
        else:
            n_outputs = self.dim + self.dim*(self.dim + 1) // 2
        self.ff = torch.nn.Sequential(
            Linear(d_x, d_model),
            ReLU(),
            Linear(d_model, d_model),
            ReLU(),
            Linear(d_model, d_model),
            ReLU(),
            Linear(d_model, n_outputs),
        )
        # eventually need to save this as an hparam if i am checkpointing models
        self.lr = lr
        self.wd = weight_decay
        self.mean_field = mean_field
        self.val_losses = []

    def forward(self, x):
        assert len(x.shape) == 2
        y = self.ff(x)
        mu = y[:, :self.dim]
        sigma = y[:, self.dim:]
        # case one: unidimensional or mean field
        if self.dim == 1:
            sigma = torch.exp(sigma)
        elif self.mean_field:
            sigma = diag(torch.exp(sigma))
        else:
            sigma = lower_tri(sigma, self.dim)
            # force diagonal entries to be positive
            sigma.diagonal(dim1=-2, dim2=-1).copy_(
                sigma.diagonal(dim1=-2,dim2=-1).exp()
            )
        return mu, sigma
    
    def training_step(self, batch, batch_idx):
        x, theta = batch
        mu, sigma = self(x)
        loss = self.gaussiannll(theta, mu, sigma)
        self.log("train_loss", loss)
        return loss

    
    def validation_step(self, batch, batch_idx):
        x, theta = batch
        assert len(theta.shape) > 1
        mu, sigma = self(x)
        loss = self.gaussiannll(theta, mu, sigma)
        self.log("val_loss", loss)
        return loss
    
    def on_validation_epoch_end(self):
        # why was this so difficult to figure out
        val_loss = self.trainer.callback_metrics["val_loss"].item()
        self.val_losses.append(val_loss)

    def gaussiannll(self, theta, mu, sigma):
        p = self.dim
        if p == 1:
            normal = Normal(mu, sigma)
            l = - normal.log_prob(theta)
        else:
            L = sigma
            mvn = MultivariateNormal(loc=mu, scale_tril=L)
            l = - mvn.log_prob(theta)

        return l.mean()
    

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
    

    def predict_step(self, x):
        # this returns standard deviation
        mu, sigma = self(x)
        return mu, sigma
    
    # TODO would it make sense to have a "sample" method?