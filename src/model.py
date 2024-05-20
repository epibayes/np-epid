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
        # assume diagonal covariance matrix
        n_outputs = self.dim * 2
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
        self.val_losses = []

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
            L = torch.diag_embed(sigma)
            mvn = MultivariateNormal(loc=mu, scale_tril=L)
            l = - mvn.log_prob(theta)

        return l.mean()
    

    def configure_optimizers(self):
        # TODO: consider swapping out for SGD with decay
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    

    def predict_step(self, x):
        # this returns standard deviation!
        mu, sigma = self(x)
        return mu, sigma