import hydra
import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from hydra.utils import instantiate
from src.utils import DataModule, save_results
from torch import device, no_grad

TOY_EXPERIMENTS = ("normal-normal", "bayes-linreg")

@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(cfg):
    gpu = device(cfg.gpu_device)
    dataset = instantiate(cfg.simulator)
    observed_data = dataset.get_observed_data()
    if cfg.train.batch_size is None:
        batch_size = cfg.simulator.n_sample
    else:
        batch_size = cfg.train.batch_size
    datamodule = DataModule(
        dataset, cfg.train.seed, batch_size, cfg.train.train_frac
        )
    model = instantiate(cfg.model, d_x=dataset.d_x, d_theta=dataset.d_theta)
    if cfg.log:
        wandb.init(reinit=False)
        logger = WandbLogger(project='crkp')
    else:
        logger = None

    if cfg.train.stop_early:
        callbacks = callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=cfg.train.patience)]
    else:
        callbacks = None
    trainer = L.Trainer(max_epochs=cfg.train.max_epochs, logger=logger,
                        devices=cfg.train.devices,
                        log_every_n_steps=cfg.train.log_freq, callbacks=callbacks,
                        fast_dev_run=cfg.fast_dev_run)

    trainer.fit(model, datamodule=datamodule)
    if cfg.model == "gdn":
        posterior_params = model.predict_step(observed_data)
        if cfg.simulator in TOY_EXPERIMENTS:
            dataset.evaluate(posterior_params)
        else:
            save_results(posterior_params, model.val_losses, cfg)
    if cfg.model == "flow":
        with no_grad():
            M = cfg.n_posterior_sample
            cfg.model.to(gpu).sample(
                (M,), dataset.get_observed_data(M).to(gpu)
            )
    wandb.finish()



    

if __name__ == "__main__":
    main()