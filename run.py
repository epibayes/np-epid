import hydra
import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from hydra.utils import instantiate

# from src.dataset import ExponentialToyDataset
from src.utils import DataModule

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
    model = instantiate(cfg.model, dataset.d_x, dataset.d_theta)
    wandb.init(reinit=False)
    logger = WandbLogger(project='crkp')
    callbacks = [ModelCheckpoint(monitor="val_loss", mode="min")]
    trainer = L.Trainer(max_epochs=cfg.train.max_epochs, logger=logger,
                        log_every_n_steps=cfg.train.log_freq, callbacks=callbacks)

    trainer.fit(model, datamodule=datamodule)

    mu, sigma = model.predict_step(observed_data)
    dataset.evaluate(mu, sigma, observed_data)
    wandb.finish()



    

if __name__ == "__main__":
    main()