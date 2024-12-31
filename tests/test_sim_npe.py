import lightning as L
from hydra.utils import instantiate
from src.utils import DataModule
from hydra import compose, initialize


def test_npe_sim_homog():
    _test("si-model")

def test_npe_sim_heterog():
    _test("si-model-het")
    
def test_npe_sim_partial():
    _test("si-model-partial")

        
        
def _test(simulator):
    with initialize(config_path="../configs"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"simulator={simulator}",
                "train.max_epochs=10",
                "simulator.n_sample=250"
            ],
        )
    dataset = instantiate(cfg.simulator)
    observed_data = dataset.get_observed_data()
    batch_size = cfg.simulator.n_sample
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
