import pytorch_lightning as pl
import torch
from torch.nn import Linear


class DensityNetwork(pl.LightningModule):
    def __init__(self, d_x, d_theta, d_model):
        super().__init__()
        # TODO: multiple components (mixture)
        # compute number of outputs
        n_mu = d_model
        if d_theta == 1:
            n_sigma = 1
        else:
            n_sigma = d_theta + d_theta * (d_theta - 1) / 2
        n_outputs = n_mu + n_sigma


        self.ff = torch.nn.Sequential(
            Linear(d_x, d_model),
            # dropout?
            torch.nn.ReLU(),
            Linear(d_model, d_model),
            torch.nn.ReLU(),
            Linear(d_model, n_outputs),
        )
