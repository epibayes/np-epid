import lightning as L
from hydra.utils import instantiate
from src.utils import MoonsDataset
from torch.utils.data import DataLoader
from hydra import compose, initialize
from src.model import RealNVP

def test_flows():
    with initialize(config_path="../configs"):
        cfg = compose(
            config_name="config",
            overrides=[
                "train.max_epochs=10",
                "model=flow"
            ],
    )
    dataset = MoonsDataset(n_sample=200, random_state=3)
    train_loader = DataLoader(dataset, batch_size=50)
    model = instantiate(cfg.model, d_x=2)
    trainer = L.Trainer(max_epochs=1)
    trainer.fit(model=model, train_dataloaders=train_loader)
    
    
    