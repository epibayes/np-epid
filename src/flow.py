import torch
import lightning as L

class BaseFlow(L.LightningModule):
    def __init__(self, lr, weight_decay, dist=None,):
        super().__init__()
        # latent distribution
        if dist is None:
            dist = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))
        # self.d = d
        self.dist = dist
        self.lr = lr
        self.wd = weight_decay

    def forward(self, X, Y=None):
        # X conditional on Y
        raise NotImplementedError()
        return Z, log_det
    
    def backward(self, Z, Y=None):
        raise NotImplementedError()
        return X
    

    def log_prob(self, Z, log_det):
        prob_Z = self.dist.log_prob(Z).sum(dim=1) # why summation?
        prob_X = prob_Z + log_det
        return prob_X

    def sample(self, sample_shape=torch.Size([1, 1]), cond_inputs=None):
        # if self.dist.loc.device != self.device:
        #     self.dist.loc = self.dist.loc.to(self.device)
        #     self.dist.scale = self.dist.scale.to(self.device)
        Z = self.dist.rsample([sample_shape[0], self.d])
        return self.inverse(Z, Y=cond_inputs)[0]
    
    def training_step(self, batch, batch_idx):
        x, theta = batch
        Z, log_det = self(x, theta)
        loss = - self.log_prob(Z, log_det)
        self.log("train_loss", loss)
        return loss

    
    def validation_step(self, batch, batch_idx):
        x, theta = batch
        Z, log_det = self(x, theta)
        loss = - self.log_prob(Z, log_det)
        self.log("val_loss", loss)
        return loss
    
    def on_validation_epoch_end(self):
        # why was this so difficult to figure out
        val_loss = self.trainer.callback_metrics["val_loss"].item()
        self.val_losses.append(val_loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)


# class SequentialFlow(GenericFlow):
#     def __init__(self, d, flows: nn.Sequential, dist=None, lr=1e-3):
#         super().__init__(d=d, dist=dist, lr=lr)
#         self.flows = flows

#     def direct(self, X, cond_inputs=None):
#         U = X
#         log_dets = torch.zeros(X.size(0), device=X.device)
#         for module in self.flows._modules.values():
#             X = U
#             U, log_dets_step = module.direct(X, cond_inputs=cond_inputs)
#             log_dets += log_dets_step
#         return U, log_dets

#     def inverse(self, U, cond_inputs=None):
#         X = U
#         log_dets = torch.zeros(U.size(0), device=U.device)
#         for module in reversed(self.flows._modules.values()):
#             U = X
#             X, log_dets_step = module.inverse(U, cond_inputs=cond_inputs)
#             log_dets += log_dets_step
#         return X, log_dets
    
# class BatchNormFlow(GenericFlow):
#     """"""

#     def __init__(self, d, momentum=0.0, eps=1e-5):
#         super().__init__(d=d)

#         self.log_gamma = nn.Parameter(torch.zeros(d))
#         self.beta = nn.Parameter(torch.zeros(d))
#         self.momentum = momentum
#         self.eps = eps

#         self.register_buffer("running_mean", torch.zeros(d))
#         self.register_buffer("running_var", torch.ones(d))

#     def direct(self, X, cond_inputs=None):
#         if self.training:
#             self.batch_mean = X.mean(0)
#             self.batch_var = (X - self.batch_mean).pow(2).mean(0) + self.eps
#             self.running_mean.mul_(self.momentum)
#             self.running_var.mul_(self.momentum)

#             self.running_mean.add_(self.batch_mean.data * (1 - self.momentum))
#             self.running_var.add_(self.batch_var.data * (1 - self.momentum))
#             mean = self.batch_mean
#             var = self.batch_var
#         else:
#             mean = self.running_mean
#             var = self.running_var
#         x_hat = (X - mean) / var.sqrt()
#         y = torch.exp(self.log_gamma) * x_hat + self.beta
#         return y, (self.log_gamma - 0.5 * torch.log(var)).sum(-1, keepdim=True)

#     def inverse(self, U, cond_inputs=None):
#         if self.training and hasattr(self, "batch_mean"):
#             mean = self.batch_mean
#             var = self.batch_var
#         else:
#             mean = self.running_mean
#             var = self.running_var
#         x_hat = (U - self.beta) / torch.exp(self.log_gamma)
#         y = x_hat * var.sqrt() + mean
#         return y, (-self.log_gamma + 0.5 * torch.log(var)).sum(-1, keepdim=True)
    
    
# class CouplingFlow(GenericFlow):
#     """"""

#     def __init__(
#         self, d, n_hidden, mask, num_cond_inputs=None, s_act="tanh", t_act="relu"
#     ):
#         super().__init__(d=d)

#         self.n_hidden = n_hidden
#         self.register_buffer("mask", mask.float())

#         activations = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}
#         s_act_func = activations[s_act]
#         t_act_func = activations[t_act]

#         if num_cond_inputs is not None:
#             total_inputs = d + num_cond_inputs
#         else:
#             total_inputs = d

#         self.scale_net = nn.Sequential(
#             nn.Linear(total_inputs, n_hidden),
#             s_act_func(),
#             nn.Linear(n_hidden, n_hidden),
#             s_act_func(),
#             nn.Linear(n_hidden, d),
#         )
#         self.translate_net = nn.Sequential(
#             nn.Linear(total_inputs, n_hidden),
#             t_act_func(),
#             nn.Linear(n_hidden, n_hidden),
#             t_act_func(),
#             nn.Linear(n_hidden, d),
#         )

#     def direct(self, X, cond_inputs=None):
#         masked_X = X * self.mask
#         if cond_inputs is not None:
#             masked_X = torch.cat([masked_X, cond_inputs], -1)
#         log_s = self.scale_net(masked_X) * (1 - self.mask)
#         t = self.translate_net(masked_X) * (1 - self.mask)
#         s = torch.exp(log_s)
#         return X * s + t, log_s.sum(-1)

#     def inverse(self, U, cond_inputs=None):
#         masked_U = U * self.mask
#         if cond_inputs is not None:
#             masked_U = torch.cat([masked_U, cond_inputs], -1)
#         log_s = self.scale_net(masked_U) * (1 - self.mask)
#         t = self.translate_net(masked_U) * (1 - self.mask)
#         s_reciprocal = torch.exp(-log_s)
#         return (U - t) * s_reciprocal, -log_s.sum(-1)


# class BatchNormFlowReversed(BatchNormFlow):
#     def __init__(self, d, momentum=0.0, eps=1e-5):
#         super().__init__(d=d, momentum=momentum, eps=1e-5)

#     def direct(self, X, cond_inputs=None):
#         return super().inverse(X, cond_inputs=cond_inputs)

#     def inverse(self, U, cond_inputs=None):
#         return super().direct(U, cond_inputs=cond_inputs)
    
# class StandardizationFlow(GenericFlow):
#     def __init__(self, mu, sigma):
#         self.d = mu.size(0)
#         super().__init__(d=self.d)
#         self.register_buffer("mu", mu)
#         self.register_buffer("sigma", sigma)

#     def direct(self, X, cond_inputs=None):
#         U = (X - self.mu) / self.sigma
#         log_det = -self.sigma.log().sum()
#         return U, log_det

#     def inverse(self, U, cond_inputs=None):
#         X = U * self.sigma + self.mu
#         log_det = self.sigma.log().sum()
#         return X, log_det


# class RealNVP(SequentialFlow):
#     """"""

#     def __init__(
#         self,
#         d,
#         n_layers,
#         n_hidden,
#         mu=None,
#         sigma=None,
#         num_cond_inputs=None,
#         reverse_batch_norm=False,
#         lr=1e-3,
#     ):
#         mask = torch.arange(0, d) % 2
#         flows = []
#         # TODO do i need to standardize?
#         # or do i already handle this in the preprocessing phase
#         if (mu is not None) and (sigma is not None):
#             flows.append(StandardizationFlow(mu, sigma))
#         if reverse_batch_norm:
#             BNF = BatchNormFlowReversed
#         else:
#             BNF = BatchNormFlow
#         for _ in range(n_layers):
#             flows.append(
#                 CouplingFlow(
#                     d, n_hidden, mask, num_cond_inputs, s_act="tanh", t_act="relu"
#                 )
#             )
#             flows.append(BNF(d))
#             mask = 1 - mask
#         super().__init__(d, nn.Sequential(*flows), lr=lr)
#         self.d = d
#         self.n_layers = n_layers
#         self.n_hidden = n_hidden