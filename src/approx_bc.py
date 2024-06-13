import numpy as np
from timeit import default_timer as timer

# frustrating i can't call this abc.py


def abc_rejection_sampler(S, epsilon, prior_sampler, simulator, 
                          x_o, max_attempts=10000, summarize=False):
    # S: total number of particles
    samples = []
    attempts = 0
    x_o = x_o.transpose(0,1)
    # x_o is shape (d_theta, d_x)
    errors = np.full(max_attempts, -1e3)
    start_time = timer()
    for s in range(S):
        accept = False
        while not accept:
            theta = np.array(prior_sampler())[0]
            x = simulator(theta, seed=attempts)
            accept, error = accept_sample(x, x_o, epsilon, summarize)
            if accept:
                samples.append(theta)
            errors[attempts] = error
            attempts += 1
            if attempts == max_attempts:
                print("Maximum attempts reached, halting")
                return np.array(samples), errors
            if not attempts % 5000:
                print(f"Attempts: {attempts:,}")
    end_time = timer()
    accept_rate = S / attempts
    
    print(f"Time lapsed: {end_time - start_time:.2f} seconds")
    print(f"With tolerance {epsilon}, acceptance rate: {accept_rate:.6f}")
    print(f"Total number of attempts: {attempts:,}")
    return np.array(samples), errors

def accept_sample(x, x_o, epsilon, summarize):
    # error should have dimension (d_theta, d_x)
    if summarize:
        x_o = x_o[:, 0]
    # w = 1 if len(x_o) > 1 else None
    v = np.array(x - x_o)
    # TODO: introduce scaling
    error = np.linalg.norm(v)
    # if not summarize:
    #     error = error.mean(w)
    accept = (error < epsilon)
    # if accept.dim() > 0:
    #     accept = accept.min()
    return accept, error