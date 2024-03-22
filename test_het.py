from src.dataset import SIModel

beta_true = []
alpha = [0.05, 0.05, 0.15]
gamma = 0.05
prior_mu = -3
prior_sigma = 1
n_zones = 1
N = 100
T = 52
K = 30

si_model = SIModel(alpha, gamma, beta_true, 
                   prior_mu, prior_sigma, n_zones, 
                   N, T, summarize=False)

x_o_raw = si_model.get_observed_data(29)