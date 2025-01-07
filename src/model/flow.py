import torch
import lightning as L
from torch.nn import ReLU, Tanh, Linear, Sequential
from torch.distributions import MultivariateNormal

class BaseFlow(L.LightningModule):
    def __init__(self, lr=1e-3, weight_decay=0):
        super().__init__()
        self.dist = None # distribution of latents
        self.lr = lr
        self.wd = weight_decay
        self.val_losses = []

    def forward(self, X, Y=None):
        # X conditional on Y
        raise NotImplementedError()
        return Z, log_det
    
    def inverse(self, Z, Y=None):
        raise NotImplementedError()
        return X
    

    def log_prob(self, Z, log_det):
        prob_Z = self.dist.log_prob(Z)
        prob_X = prob_Z + log_det
        return prob_X

    def sample(self, sample_size, cond_inputs=None):
        # if self.dist.loc.device != self.device:
        #     self.dist.loc = self.dist.loc.to(self.device)
        #     self.dist.scale = self.dist.scale.to(self.device)
        Z = self.dist.rsample(sample_size)
        return self.inverse(Z, Y=cond_inputs)
    
    def training_step(self, batch, batch_idx):
        x, theta = batch
        Z, log_det = self(x, theta)
        loss = - self.log_prob(Z, log_det).mean()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    
    def validation_step(self, batch, batch_idx):
        x, theta = batch
        Z, log_det = self(x, theta)
        loss = - self.log_prob(Z, log_det).mean()
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
        # why was this so difficult to figure out
        val_loss = self.trainer.callback_metrics["val_loss"].item()
        self.val_losses.append(val_loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
    
class CouplingFlow(BaseFlow):

    def __init__(
        self, d_x, d_model, mask, d_cond=0
    ):
        super().__init__()
        self.d_model = d_model
        self.register_buffer("mask", mask.float())

        s_act_func = Tanh # why Tanh though...
        t_act_func = ReLU

        total_inputs = d_x + d_cond


        self.scale_net = Sequential(
            Linear(total_inputs, d_model),
            s_act_func(),
            Linear(d_model, d_model),
            s_act_func(),
            Linear(d_model, d_x),
        )
        self.translate_net = Sequential(
            Linear(total_inputs, d_model),
            t_act_func(),
            Linear(d_model, d_model),
            t_act_func(),
            Linear(d_model, d_x),
        )

    def forward(self, X, Y=None):
        masked_X = X * self.mask
        if Y is not None:
            masked_X = torch.cat([masked_X, Y], -1)
        log_s = self.scale_net(masked_X) * (1 - self.mask)
        t = self.translate_net(masked_X) * (1 - self.mask)
        s = torch.exp(log_s)
        return X * s + t, log_s.sum(-1)

    def inverse(self, Z, Y=None):
        masked_Z = Z * self.mask
        if Y is not None:
            masked_Z = torch.cat([masked_Z, Y], -1)
            # compute s and t on the masked portion of the output (same as input)
        log_s = self.scale_net(masked_Z) * (1 - self.mask)
        t = self.translate_net(masked_Z) * (1 - self.mask)
        s_reciprocal = torch.exp(-log_s)
        return (Z - t) * s_reciprocal
    
    
class RealNVP(BaseFlow):

    def __init__(
        self,
        d_x,
        n_layers,
        d_model,
        lr=1e-3,
        weight_decay=0,
        d_cond=0
    ):
        super().__init__(lr, weight_decay)
        self.register_buffer("loc", torch.zeros(d_x, device=self.device))
        self.register_buffer("cov", torch.eye(d_x, device=self.device))
        
        # self.dist = MultivariateNormal(torch.zeros(d_x), torch.eye(d_x))
        mask = torch.arange(0, d_x) % 2 # alternating bit mask
        # assign as attribute to register parameters correctly
        self.flows = torch.nn.ModuleList()
        # define target, latent distribution
        # if (mu is not None) and (sigma is not None):
        #     flows.append(StandardizationFlow(mu, sigma))
        # if reverse_batch_norm:
        #     BNF = BatchNormFlowReversed
        # else:
        #     BNF = BatchNormFlow
        for _ in range(n_layers):
            self.flows.append(
                CouplingFlow(d_x, d_model, mask, d_cond)
            )
            # flows.append(BNF(d))
            mask = 1 - mask # flip the bit mask
            
    def on_fit_start(self):
        # necessary for getting the latent distribution on the same device as the model
        self.dist = MultivariateNormal(self.loc, self.cov)
        
        
    def forward(self, X, Y=None):
        log_det_sum = torch.zeros(X.size(0), device=self.device) # device=U.device
        for module in self.flows:
            Z, log_det = module.forward(X, Y)
            log_det_sum += log_det
            X = Z
        return Z, log_det_sum

    def inverse(self, Z, Y=None):
        for module in reversed(self.flows):
            X = module.inverse(Z, Y=None)
            Z = X
        return X
    
    def set_prior(self):

        return MultivariateNormal(self.loc, self.cov)
    