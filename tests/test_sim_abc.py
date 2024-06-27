from src.dataset import SIModel
from src.approx_bc import abc_rejection_sampler

alpha = 0.1
gamma = 0.05
T = 52
K = 30
observed_seed=29
N = 100
S = 10
epsilon = 10

def test_abc_sim_homog():
    beta_true = .15
    prior_mu = -3
    prior_sigma = 1
    het = False
    flatten = False
    summarize = False
    
    _test(beta_true, prior_mu, prior_sigma, het, summarize, flatten)

def test_abc_sim_homog_summ():
    beta_true = .15
    prior_mu = -3
    prior_sigma = 1
    het = False
    flatten = False
    summarize = True
    
    _test(beta_true, prior_mu, prior_sigma, het, summarize, flatten)
    
def test_abc_sim_heterog():
    beta_true = [0.075, 0.3, 0.2, 0.4, .2, .3, .25, .5]
    prior_mu = [-3, -2, -2, -2, -2, -2, -2, -1]
    prior_sigma = [1 for _ in range(8)]
    het = True
    summarize = False
    flatten = False
    
    _test(beta_true, prior_mu, prior_sigma, het, summarize, flatten)
    

def test_abc_sim_heterog_summ():
    beta_true = [0.075, 0.3, 0.2, 0.4, .2, .3, .25, .5]
    prior_mu = [-3, -2, -2, -2, -2, -2, -2, -1]
    prior_sigma = [1 for _ in range(8)]
    het = True
    summarize = True
    flatten = False
    
    _test(beta_true, prior_mu, prior_sigma, het, summarize, flatten)
    
    
def _test(beta_true, prior_mu, prior_sigma, het, summarize, flatten):
    si_model = SIModel(alpha, gamma, beta_true, het, 
            prior_mu, prior_sigma,
            N, T, observed_seed=observed_seed,
            summarize=summarize,
            flatten=flatten)

    x_o = si_model.get_observed_data()
    prior_sampler = lambda: si_model.sample_logbeta(1)
    simulator = lambda theta, seed: si_model.SI_simulator(theta, seed)

    abc_rejection_sampler(
        S, epsilon, prior_sampler, simulator, x_o, max_attempts=500,
        summarize=summarize
            )