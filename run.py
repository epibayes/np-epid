import hydra

import numpy as np
import pandas as pd
import lightning as L

from lightning.pytorch.loggers import WandbLogger
from src.dataset import ExponentialToyDataset
from src.utils import DataModule
from src.model import GaussianDensityNetwork

@hydra.main(config_path=".", config_name="config.yaml", version_base=None)
def main(cfg=None):
    data_cfg = cfg[cfg.experiment]
    # todo: switch over to instantiate
    if cfg.experiment == "toy-exponential":
        dataset = ExponentialToyDataset(
            data_cfg.n_obs, data_cfg.n_sample, data_cfg.shape, data_cfg.scale,
            data_cfg.random_state
        )

    datamodule = DataModule(
        dataset, cfg.train.seed, cfg.train.batch_size, cfg.train.train_frac
        )
    # switch over to instantiate if this gets too clunky
    model = GaussianDensityNetwork(
        dataset.d_x, dataset.d_theta, cfg.model.d_model, cfg.model.lr
        )

    logger = WandbLogger(project='crkp')
    trainer = L.Trainer(max_epochs=cfg.train.max_epochs, logger=logger,
                        log_every_n_steps=cfg.train.log_freq)

    trainer.fit(model, datamodule=datamodule)

    # predict step?
    # generate/load "true" data & feed it through the trained nn



    

if __name__ == "__main__":
    main()