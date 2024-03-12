import numpy as np
from timeit import default_timer as timer

# frustrating i can't call this abc.py


def abc_rejection_sampler(S, epsilon, sample_prior, simulate_data, 
                          x_o, max_attempts=10000):
    # S: total number of particles
    samples = []
    attempts = 0
    errors = np.full((max_attempts,), -1e3)
    start_time = timer()
    for s in range(S):
        accept = False
        while not accept:
            theta = sample_prior()
            x = simulate_data(theta, seed=attempts)
            accept, error = accept_sample(x, x_o, epsilon)
            if accept:
                samples.append(theta)
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

def accept_sample(x, x_o, epsilon):
    if x_o.shape[0] == 1:
        error = ((x - x_o)**2).mean()
        accept = (error < epsilon)
    else:
        error = ((x - x_o)**2).mean(1).max()
        accept = (error < epsilon)
    return accept, error