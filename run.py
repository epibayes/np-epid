import numpy as np
import pandas as pd

from src.simulator import CRKPTransmissionSimulator


# TODO: put this into a config
PATH = "Volumes/umms-esnitkin/Project_KPC_LTACH/Analysis/LTACH_transmission_modeling/data"

def main():

    simulator = CRKPTransmissionSimulator(PATH)
    simulator.load_data()


    # transmission rate
    beta = np.exp(
        np.random.normal(scale=3, size=8)
    )

    simulator = CRKPTransmissionSimulator
    return

if __name__ == "__main__":
    main()