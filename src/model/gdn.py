import lightning as L
import torch
from torch.nn import Linear, ReLU, Embedding
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

from src.utils import lower_tri, diag
import math



class GaussianDensityNetworkBase(L.LightningModule):
    def __init__(self, d_theta, lr, weight_decay,
                 mean_field):
        super().__init__()
        self.estimator = "gdn"
        # compute number of outputs
        self.dim = d_theta
        # assume diagonal covariance matrix
        if mean_field:
            self.n_outputs = self.dim * 2
        else:
            self.n_outputs = self.dim + self.dim*(self.dim + 1) // 2
        # eventually need to save this as an hparam if i am checkpointing models
        self.lr = lr
        self.wd = weight_decay
        self.mean_field = mean_field
        self.val_losses = []
        
    def encoder(self, x):
        raise NotImplementedError

    def forward(self, x):
        y = self.encoder(x)
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


class GaussianDensityNetwork(GaussianDensityNetworkBase):
    def __init__(self, d_x, d_theta, d_model, lr, weight_decay,
                 mean_field, first_dim=None):
        super().__init__(d_theta, lr, weight_decay,
                 mean_field)

        
        if first_dim is None: first_dim = d_x

        self.ff = torch.nn.Sequential(
            Linear(first_dim , d_model),
            ReLU(),
            Linear(d_model, d_model),
            ReLU(),
            Linear(d_model, d_model),
            ReLU(),
            Linear(d_model, self.n_outputs),
        )
        # eventually need to save this as an hparam if i am checkpointing models
        self.lr = lr
        self.wd = weight_decay
        self.mean_field = mean_field
        self.val_losses = []

        
    def encoder(self, x):
        return self.ff(x)
    
class SummaryGDN(GaussianDensityNetwork):
    def __init__(self, d_x, d_theta, d_model, lr, weight_decay,
                 mean_field, d_summ):
        
        first_dim = d_x * d_summ
        super().__init__(d_x, d_theta, d_model, lr, weight_decay, mean_field,
                         first_dim)
        
        self.summarize = torch.nn.Sequential(
                Linear(1, d_summ),
                ReLU(),
                Linear(d_summ, d_summ*2),
                ReLU(),
                Linear(d_summ*2, d_summ)
            )
        
        self.register_buffer("floor_trace", None)
        self.register_buffer("room_trace", None)
        self.register_buffer("mask", None)
        # self.floor_trace = None
        # self.room_trace = None
        # self.mask = None
        self.d_summ = d_summ
        
        
    def load_traces(self, floor_trace, room_trace):
        self.register_buffer("floor_trace", torch.tensor(floor_trace, device=self.device))
        self.register_buffer("room_trace", torch.tensor(room_trace, device=self.device))
        
        # create embedding layers
        self.floor_embedding = Embedding(self.floor_trace.max() + 1, 3)
        self.room_embedding = Embedding(self.room_trace.max() + 1, 8) 
        
        self.summarize = torch.nn.Sequential(
                Linear(12, self.d_summ),
                ReLU(),
                Linear(self.d_summ, self.d_summ),
                ReLU(),
                Linear(self.d_summ, self.d_summ),
            )
    

    def load_mask(self, mask):
        self.register_buffer("mask", torch.tensor(mask, device=self.device))
            
    
    def encoder(self, x):
        x = torch.nan_to_num(x).unsqueeze(-1)
        # new idea: torch.where(self.floor_trace = f, x, 0)
        if (self.floor_trace is not None and self.room_trace is not None):
            m = x.shape[0]
            f = self.floor_embedding(self.floor_trace).expand(m, -1, -1, -1)
            r = self.room_embedding(self.room_trace).expand(m, -1, -1, -1)
            x = torch.cat([x, f, r], -1)
        
        x = self.summarize(x)
        if self.mask is not None:
            w = self.mask.unsqueeze(-1)
            x = x * w
        # pool over observations
        # x = x.sum(1)
        x = x.mean(1) # mean pooling helps prevent exploding gradient
        x = x.flatten(1, 2)
        return self.ff(x)

class GaussianDensityRNN(GaussianDensityNetworkBase):
    def __init__(self, d_x, d_theta, d_model, lr, weight_decay,
                mean_field, n_layers, dropout):
        super().__init__(d_theta, lr, weight_decay,
                mean_field)
        
        self.LSTM = torch.nn.LSTM(
            input_size=d_x[1], hidden_size=d_model, num_layers=n_layers,
            dropout=dropout, batch_first=True, 
        )
        self.to_output = torch.nn.Sequential(
            ReLU(),
            Linear(d_model, self.n_outputs)
        )
        
    def encoder(self, x):
        y, _ = self.LSTM(x)
        y = y.mean(dim=1)
        return self.to_output(y)
    
    
class GaussianDensityTransformer(GaussianDensityNetworkBase):
    def __init__(self, d_x, d_theta, d_model, lr, weight_decay,
                mean_field, n_heads, dropout, n_blocks):
        super().__init__(d_theta, lr, weight_decay,
                mean_field)
        
        self.pos_encode = PositionalEncoding(d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=d_model*4 # it's a heuristic idk
        )
        norm = torch.nn.LayerNorm(d_model)
        assert type(d_x) == torch.Size
        self.embed = Linear(d_x[1], d_model)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, n_blocks, norm)
        self.to_output = torch.nn.Sequential(
            ReLU(),
            Linear(d_model, self.n_outputs),
        )
        
    def encoder(self, x):
        x = self.embed(x)
        x = self.pos_encode(x)
        y = self.transformer(x)
        # pooling operation
        # averaging makes the most intuitive sense imo
        y = y.mean(dim=1)
        return self.to_output(y)
        
        
        
class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, dropout=0.0, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return self.dropout(x)