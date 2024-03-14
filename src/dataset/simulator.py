from torch.utils.data import Dataset


class Simulator(Dataset):
    def __init__(self, n_sample, random_state):
        self.n_sample = None
        self.data = None
        self.theta = None
        self.d_x = None
        self.d_theta = None
        self.random_state = None

    def __len__(self):
        return self.n_sample
    
    def __getitem__(self, index):
        return self.data[index], self.theta[index]
    
    def simulate_data(self):
        raise NotImplementedError
    
    def get_observed_data(self):
        raise NotImplementedError
    
    def evaluate(self, posterior_params):
        pass
