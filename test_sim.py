from src.dataset import SIModel
from src.approx_bc import abc_rejection_sampler

beta_true = .15
alpha = 0.1
gamma = 0.05
prior_mu = -3
prior_sigma = 1
n_zones = 1
N = 100
T = 52
K = 30
summarize = False

si_model = SIModel(alpha, gamma, beta_true, n_zones, 
                   prior_mu, prior_sigma,
                   N, T, summarize)

x_o = si_model.get_observed_data(29)
if not summarize:
    x_o = x_o.transpose(0, 1)
prior_sampler = lambda: si_model.sample_logbeta(1)
simulator = lambda theta, seed: si_model.SI_simulator(theta, seed)

S = 100
epsilon = 0.01
posterior_sample, errors = abc_rejection_sampler(
    S, epsilon, prior_sampler, simulator, x_o, max_attempts=500,
      summarize=summarize
    )
1/0
