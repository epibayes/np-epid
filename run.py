import hydra

import numpy as np
import pandas as pd

from src.simulator import CRKPTransmissionSimulator


@hydra.main(config_path=".", config_name="config.yaml", version_base=None)
def main(cfg=None):

    # simulator = CRKPTransmissionSimulator(PATH, sigma=3)
    # simulator.load_data()


    # # transmission rate
    # beta = np.exp(
    #     np.random.normal(scale=3, size=8)
    # )

    # simulator._simulate(beta)

if __name__ == "__main__":
    main()