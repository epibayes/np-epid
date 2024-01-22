import hydra
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from hydra.utils import instantiate

# from src.dataset import ExponentialToyDataset
from src.utils import DataModule
from src.model import GaussianDensityNetwork

@hydra.main(config_path=".", config_name="config.yaml", version_base=None)
def main(cfg=None):
    data_cfg = cfg[cfg.experiment]
    # todo: switch over to instantiate
    dataset = instantiate(data_cfg)
    observed_data = dataset.get_observed_data()


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

    mu, sigma = model.predict_step(observed_data)
    dataset.evaluate(mu, sigma, observed_data)



    

if __name__ == "__main__":
    main()