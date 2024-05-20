import hydra
import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from hydra.utils import instantiate
from src.utils import DataModule, save_results

TOY_EXPERIMENTS = ("test-dataset", "normal-normal", "bayes-linreg")

@hydra.main(config_path=".", config_name="config.yaml", version_base=None)
def main(cfg=None):
    data_cfg = cfg[cfg.experiment]
    dataset = instantiate(data_cfg)
    observed_data = dataset.get_observed_data(cfg["observed_seed"])
    datamodule = DataModule(
        dataset, cfg.train.seed, cfg.train.batch_size, cfg.train.train_frac
        )
    model = instantiate(cfg.model, dataset.d_x, dataset.d_theta)
    wandb.init(reinit=False)
    logger = WandbLogger(project='crkp')
    callbacks = [ModelCheckpoint(monitor="val_loss", mode="min")]
    trainer = L.Trainer(max_epochs=cfg.train.max_epochs, logger=logger,
                        log_every_n_steps=cfg.train.log_freq, callbacks=callbacks)

    trainer.fit(model, datamodule=datamodule)

    posterior_params = model.predict_step(observed_data)
    if cfg.experiment in TOY_EXPERIMENTS:
        dataset.evaluate(posterior_params)
    else:
        multidim = (dataset.d_theta > 1)
        save_results(posterior_params, model.val_losses, cfg, multidim)
    wandb.finish()



    

if __name__ == "__main__":
    main()