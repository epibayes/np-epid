import numpy as np
from timeit import default_timer as timer

# frustrating i can't call this abc.py


def abc_rejection_sampler(S, epsilon, prior_sampler, simulator, 
                          x_o, max_attempts=10000, summarize=False,
                          print_every=5000, error_scaling = None):
    # S: total number of particles
    samples = []
    attempts = 0
    # x_o is shape (d_theta, d_x)
    errors = np.full(max_attempts, -1e3)
    start_time = timer()
    for s in range(S):
        accept = False
        while not accept:
            theta = np.array(prior_sampler())[0]
            x = simulator(theta, seed=attempts)
            accept, error = accept_sample(x, x_o, epsilon, summarize, error_scaling)
            if accept:
                samples.append(theta)
            errors[attempts] = error
            attempts += 1
            if attempts == max_attempts:
                print("Maximum attempts reached, halting")
                return np.array(samples), errors
            if not attempts % print_every:
                print(f"Attempts: {attempts:,}")
    end_time = timer()
    accept_rate = S / attempts
    
    print(f"Time lapsed: {end_time - start_time:.2f} seconds")
    print(f"With tolerance {epsilon}, acceptance rate: {accept_rate:.6f}")
    print(f"Total number of attempts: {attempts:,}")
    return np.array(samples), errors


def abc_rejection_sampler2(S, prior_sampler, simulator, 
                          x_o, summarize=False,
                          error_scaling = None):
        # S: total number of particles
    samples = np.empty(S)
    # x_o is shape (d_theta, d_x)
    errors = np.full(S, -1e3)
    seed = 0
    start_time = timer()
    for s in range(S):
        theta = np.array(prior_sampler())[0]
        x = simulator(theta, seed=seed)
        # this is gimmicky
        _, error = accept_sample(x, x_o, 1e5, summarize, error_scaling)
        samples[s] = theta
        errors[s] = error
        seed += 1
    return samples, errors
    

def accept_sample(x, x_o, epsilon, summarize, lam):
    if summarize:
        x_o = x_o[0]
    v = np.array(x - x_o)
    if lam is not None:
        v = v * lam.reshape(-1, 1)
    error = np.linalg.norm(v)
    accept = (error < epsilon)

    return accept, error