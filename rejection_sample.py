import numpy as np
from src.utils import contact_matrix, simulator, nll, x_loglikelihood
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
      

def main():
    beta_true = np.array([0.05, .1, .2, .3, .4, .5, 5])
    alpha = 0.1
    gamma = 0.02
    heterogeneous = True
    N = 300 # consider increasing
    T = 52
    K = 30
    
    F = np.arange(N) % 5
    R = np.arange(N) % (N // 2)
    fC = contact_matrix(F)
    rC = contact_matrix(R)
    
    X_o = simulator(alpha, beta_true, gamma, N, T, seed=31, het=heterogeneous)
    
    ml = minimize(
        nll, x0 = beta_true, args = (alpha, gamma, N, T, X_o, True),
        bounds = [(0.0, None) for _ in range(7)], tol=0.001
    )
    
    prior_mu = np.array([-3, -1.5, -1.5, -1.5, -1.5, -1.5, 1])

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
    
    np.save("posterior_sample2", sample)

if __name__ == "__main__":
    main()