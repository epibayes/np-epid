import numpy as np
from timeit import default_timer as timer
#TODO: inheritance magic to wrap this as a torch dataset

class SIS_Simulator():
    def __init__(self, alpha, gamma, beta_true, 
                 prior_mu, prior_sigma, N=100, T=52):
        self.alpha = alpha # baseline proportion infected in pop
        self.gamma = gamma # discharge rate
        self.N = N
        self.T = T
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        # observed data
        self.x_o = self.simulate_SIS(beta_true, seed=29)

    def simulate_SIS(self, beta, seed=None):
        # beta is infection rate
        if seed is not None:
            np.random.seed(seed)
        A  = np.empty((self.N, self.T + 1))
        # seed initial infections
        A[:, 0] = np.random.binomial(1, self.alpha, 100)

        for t in range(1, self.T+1):
            status = A[:, t-1]
            # TODO: vectorize this
            for j in range(self.N):
                # simulate probability of infection
                if status[j] == 0:
                    hazard_j = status.sum() * beta / self.N
                    p_j = 1 - np.exp(-hazard_j)
                    # simulate 
                    A[j, t] = np.random.binomial(1, p_j)
                else:
                    A[j, t] = 1

                # calc probability of discharge
                discharge = np.random.binomial(1, self.gamma)
                if discharge: 
                    A[j, t] = np.random.binomial(1, self.alpha)

        return A.mean(0) # proportion of infecteds at each time step
    
    def sample_beta(self):
        return np.exp(np.random.normal(self.prior_mu, self.prior_sigma))
    
    def abc_rejection_sampler(self, S, epsilon, max_attempts=10000):
        accepts = np.empty(S)
        attempts = 0
        errors = np.full((max_attempts,), -1e3)
        start_time = timer()
        for s in range(S):
            error = epsilon + 1
            while error >= epsilon:
                beta = self.sample_beta()
                x = self.simulate_SIS(beta)
                error = ((x - self.x_o)**2).mean()
                if error < epsilon:
                    accepts[s] = beta
                errors[attempts] = error
                attempts += 1
                if attempts == max_attempts:
                    print("Maximum attempts reached, halting")
                    return None, None
        end_time = timer()
        accept_rate = S / attempts
        
        print(f"Time lapsed: {end_time - start_time:.2f} seconds")
        print(f"With tolerance {epsilon}, acceptance rate: {accept_rate:.6f}")
        return accepts, errors

                
    
