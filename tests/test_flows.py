import lightning as L
from hydra.utils import instantiate
from src.utils import MoonsDataset
from torch.utils.data import DataLoader, random_split
from hydra import compose, initialize
from torch import device, no_grad

def test_flow():
    with initialize(config_path="../configs"):
        cfg = compose(
            config_name="config",
            overrides=[
                "train.max_epochs=10",
                "model=flow"
            ],
    )
    
    N = 500
    a = int(N * 0.8)
    b = N - a
    dataset = MoonsDataset(n_sample=N, random_state=3)
    train_data, val_data = random_split(dataset, [a, b])
    train_loader = DataLoader(train_data, batch_size=100)
    val_loader = DataLoader(val_data, batch_size=b)
    model = instantiate(cfg.model, d_x=2)
    trainer = L.Trainer(max_epochs=1, log_every_n_steps=4)
    trainer.fit(model, train_loader, val_loader)
    
    
def test_conditional_flow():
    mps_device = device("mps")
    with initialize(config_path="../configs"):
        cfg = compose(
            config_name="config",
            overrides=[
                "train.max_epochs=10",
                "model=flow",
                "simulator=two-moons",
                "simulator.n_sample=1000"
            ],
        )
    N = cfg.simulator.n_sample
    M = cfg.n_posterior_sample
    a = int(N * 0.8)
    b = N - a
    dataset = instantiate(cfg.simulator)
    train_data, val_data = random_split(dataset, [a, b])
    train_loader = DataLoader(train_data, batch_size=100)
    val_loader = DataLoader(val_data, batch_size=b)
    model = instantiate(cfg.model, d_x=2, d_theta=2)
    trainer = L.Trainer(max_epochs=1, log_every_n_steps=4)
    trainer.fit(model, train_loader, val_loader)
    with no_grad():
        model.to(mps_device).sample(M, dataset.get_observed_data())
    
    
    