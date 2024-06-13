import hydra
import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from hydra.utils import instantiate
from src.utils import DataModule, save_results

TOY_EXPERIMENTS = ("test-dataset", "normal-normal", "bayes-linreg")

@hydra.main(config_path=".", config_name="config.yaml", version_base=None)
def main(cfg=None):
    data_cfg = cfg[cfg.experiment]
    dataset = instantiate(data_cfg)
    observed_data = dataset.get_observed_data()
    if cfg.train.batch_size is None:
        batch_size = data_cfg.n_sample
    else:
        batch_size = cfg.train.batch_size
    datamodule = DataModule(
        dataset, cfg.train.seed, batch_size, cfg.train.train_frac
        )
    model = instantiate(cfg.model, dataset.d_x, dataset.d_theta)
    wandb.init(reinit=False)
    logger = WandbLogger(project='crkp')
    # callbacks = [ModelCheckpoint(monitor="val_loss", mode="min")]
    # callbacks = None
    if cfg.train.stop_early:
        callbacks = callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=cfg.train.patience)]
    else:
        callbacks = None
    trainer = L.Trainer(max_epochs=cfg.train.max_epochs, logger=logger,
                        devices=cfg.train.devices,
                        log_every_n_steps=cfg.train.log_freq, callbacks=callbacks)

    trainer.fit(model, datamodule=datamodule)

    posterior_params = model.predict_step(observed_data)
    if cfg.experiment in TOY_EXPERIMENTS:
        dataset.evaluate(posterior_params)
    else:
        multidim = (dataset.d_theta > 1)
        save_results(posterior_params, model.val_losses, cfg, multidim)
    # TODO: add logic for real data
    wandb.finish()



    

if __name__ == "__main__":
    main()