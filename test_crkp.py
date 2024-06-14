from src.dataset import CRKPTransmissionSimulator
from src.approx_bc import abc_rejection_sampler

path = "/Volumes/umms-esnitkin/Project_KPC_LTACH/Analysis/LTACH_transmission_modeling/preprocessed"
prior_mu = -2
prior_sigma = 1
summarize = False
model = CRKPTransmissionSimulator(path, prior_mu, prior_sigma)

x_o = model.get_observed_data()
# this is dumb
if not summarize:
    x_o = x_o.transpose(0, 1)

prior_sampler = lambda: model.sample_logbeta(1)
simulator = lambda theta, seed: model.CRKP_simulator(theta, seed)

S = 100
epsilon = 0.5
posterior_sample, errors = abc_rejection_sampler(
    S, epsilon, prior_sampler, simulator, x_o, max_attempts=1000,
    summarize=summarize
)
1/0