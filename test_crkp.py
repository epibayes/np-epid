from src.dataset import CRKPTransmissionSimulator
from src.approx_bc import abc_rejection_sampler
import numpy as np

path = "/Volumes/umms-esnitkin/Project_KPC_LTACH/Analysis/LTACH_transmission_modeling/preprocessed/resampled"
prior_mu = 0
prior_sigma = 1
hetero = True
if hetero:
    prior_mu = np.zeros(8)
    prior_sigma = np.ones(8)

model = CRKPTransmissionSimulator(path, prior_mu, prior_sigma,
                                heterogeneous=hetero)

x_o = model.get_observed_data()
# this is dumb

prior_sampler = lambda: model.sample_logbeta(1)
simulator = lambda theta, seed: model.CRKP_simulator(theta, seed)

S = 100
epsilon = 0.5
posterior_sample, errors = abc_rejection_sampler(
    S, epsilon, prior_sampler, simulator, x_o, max_attempts=100,
)
1/0