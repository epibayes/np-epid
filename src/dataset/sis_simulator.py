import numpy as np
from timeit import default_timer as timer
#TODO: inheritance magic to wrap this as a torch dataset



class SIS_Simulator():
    def __init__(self, alpha, gamma, beta_true, 
                 prior_mu, prior_sigma, n_zones = 1, N=100, T=52):
        self.alpha = alpha # baseline proportion infected in pop
        self.gamma = gamma # discharge rate
        self.N = N
        self.T = T
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        # observed data
        self.n_zones = n_zones
        self.x_o = self.simulate_SIS(beta_true, seed=29)

    def simulate_SIS(self, beta, seed=None):
        # beta is infection rate
        if type(beta) is float:
            beta = np.array((beta, 0))
        else:
            beta = np.array(beta)

        assert len(beta) == self.n_zones + 1
    
        if seed is not None:
            np.random.seed(seed)
        A  = np.empty((self.N, self.T + 1))
        # assign zones at random
        Z = np.random.choice(np.arange(self.n_zones), self.N)
        # seed initial infections
        A[:, 0] = np.random.binomial(1, self.alpha, self.N)

        for t in range(1, self.T+1):
            I = A[:, t-1]
            # components dependent on individual covariates
            hazard = I.sum() * beta[0] * np.ones(self.N)
            if self.n_zones > 1:
                Zx, Zy = np.meshgrid(Z, Z)
                # generate contact matrix: each row i indicates who shares a zone with patient i
                C = (Zx == Zy).astype(int)
                hazard += (C * I).sum(1) * beta[Z+1]
                # TODO: introduce room-level risk
            p = 1 - np.exp(-hazard / self.N)
            A[:,t] = np.where(I, np.ones(self.N), np.random.binomial(1, p, self.N))
            discharge = np.random.binomial(1, self.gamma, size=self.N)
            A[:,t] = np.where(discharge, np.random.binomial(1, self.alpha, self.N), A[:, t])

        if self.n_zones == 1:
            return A.mean(0) # proportion of infecteds at each time step
        else:
            zone_counts = [A[Z == i].mean(0) for i in range(self.n_zones)]
            return np.array([A.mean(0)] + zone_counts)
    
    def sample_beta(self):
        if np.isscalar(self.prior_mu) and self.n_zones > 1:
            log_beta = np.random.normal(self.prior_mu, self.prior_sigma, self.n_zones + 1)
        else:
            log_beta = np.random.normal(self.prior_mu, self.prior_sigma)
        return np.exp(log_beta)

    
    def abc_rejection_sampler(self, S, epsilon, max_attempts=10000):
        # S: total number of particles
        samples = []
        attempts = 0
        errors = np.full((max_attempts,), -1e3)
        start_time = timer()
        for s in range(S):
            accept = False
            while not accept:
                beta = self.sample_beta()
                x = self.simulate_SIS(beta, seed=attempts)
                accept, error = self.accept_sample(x, epsilon)
                if accept:
                    samples.append(beta)
                errors[attempts] = error
                attempts += 1
                if attempts == max_attempts:
                    print("Maximum attempts reached, halting")
                    return np.array(samples), errors
        end_time = timer()
        accept_rate = S / attempts
        
        print(f"Time lapsed: {end_time - start_time:.2f} seconds")
        print(f"With tolerance {epsilon}, acceptance rate: {accept_rate:.6f}")
        print(f"Total number of attempts: {attempts:,}")
        return np.array(samples), errors
    
    def accept_sample(self, x, epsilon):
        # x is the simulated data
        if self.n_zones == 1:
            error = ((x - self.x_o)**2).mean()
            accept = (error < epsilon)
        else:
            error = ((x - self.x_o)**2).mean(1).max()
            accept = (error < epsilon)
        return accept, error
