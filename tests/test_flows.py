import lightning as L
from hydra.utils import instantiate
from src.utils import MoonsDataset
from torch.utils.data import DataLoader, random_split
from hydra import compose, initialize

def test_flows():
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
    
    
    