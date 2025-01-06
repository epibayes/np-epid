import numpy as np
from src.utils import simulator, nll, x_loglikelihood
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
      
# this script won't run unless moved out of src/
# may need to append sys path
def main():
    beta_true = [.05, .02, .04, .06, .08, .1, .05]
    alpha = 0.1
    gamma = 0.05
    heterogeneous = True
    N = 300 # consider increasing
    T = 52
    
    X_o = simulator(alpha, beta_true, gamma, N, T, seed=31, het=heterogeneous)
    
    ml = minimize(
        nll, x0 = beta_true, args = (alpha, gamma, N, T, X_o, True),
        bounds = [(0.0, None) for _ in range(7)], tol=0.001
    )
    
    prior_mu = -3

    S = 100
    M = - ml.fun
    sample = np.empty((S, 7))
    attempts = 0
    np.random.seed(4)
    for s in range(S):
        accept = False
        while not accept:
            logbeta = multivariate_normal(prior_mu).rvs()
            attempts += 1
            u = np.random.uniform(0,1)
            if np.log(u) < x_loglikelihood(np.exp(logbeta), alpha, gamma, N, T, X_o, True) - M:
                accept = True
                sample[s] = logbeta
            if attempts % 10000 == 0:
                print(attempts)
        print(s)
    
    print(attempts)
    
    np.save("posterior_sample", sample)

if __name__ == "__main__":
    main()