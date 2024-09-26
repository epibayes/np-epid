import lightning as L
from hydra.utils import instantiate
from src.utils import DataModule
from hydra import compose, initialize

def test():
    with initialize(config_path=".."):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=crkp",
                "train.max_epochs=10",
                "crkp.n_sample=100"
            ],
        )
    data_cfg = cfg[cfg.experiment]
    dataset = instantiate(data_cfg)
    observed_data = dataset.get_observed_data()
    batch_size = data_cfg.n_sample
    datamodule = DataModule(
        dataset, cfg.train.seed, batch_size, cfg.train.train_frac
        )
    model = instantiate(cfg.model, dataset.d_x, dataset.d_theta)
    callbacks = None
    trainer = L.Trainer(max_epochs=cfg.train.max_epochs,
                        devices=cfg.train.devices, callbacks=callbacks,
                        detect_anomaly=True)

    trainer.fit(model, datamodule=datamodule)

    model.predict_step(observed_data)  