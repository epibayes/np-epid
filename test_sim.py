from src.dataset import SIModel
from src.approx_bc import abc_rejection_sampler

het = False

if het:
    N = 300
    beta_true = [0.075, 0.3, 0.2, 0.4, .2, .3, .25, .5]
    prior_mu = [-3, -2, -2, -2, -2, -2, -2, -1]
    prior_sigma = [1 for _ in range(8)]
else:
    N = 100
    beta_true = .15
    prior_mu = -3
    prior_sigma = 1

alpha = 0.1
gamma = 0.05
T = 52 # pick a higher T?
K = 30
summarize = False
observed_seed=29

si_model = SIModel(alpha, gamma, beta_true, het, 
                   prior_mu, prior_sigma,
                   N, T, summarize, observed_seed,
                   flatten=False)

x_o = si_model.get_observed_data()
# if not summarize:
#     x_o = x_o.transpose(0, 1)
prior_sampler = lambda: si_model.sample_logbeta(1)
simulator = lambda theta, seed: si_model.SI_simulator(theta, seed)

S = 100
epsilon = 10
posterior_sample, errors = abc_rejection_sampler(
    S, epsilon, prior_sampler, simulator, x_o, max_attempts=500,
      summarize=summarize
    )
1/0
