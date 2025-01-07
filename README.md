## Neural Posterior Estimation for Stochastic Epidemic Modeling

[![arXiv](https://img.shields.io/badge/arXiv-<2412.12967>-<COLOR>.svg)](https://arxiv.org/abs/2412.12967)

Code for calibrating stochastic infectious disease models to data through simulation-based inference and deep learning.

### How to Use

Dependencies are managed through Conda and Poetry. To create a Conda environment with Pytorch, run `conda env create --name envname --file=environments.yml`. You can install all other necessary Python packages with `poetry install`.

This repository uses Lightning to improve code readability and modularity and Hydra to manage configurations for deep learning experiments. For example, `python -m run.py simulator=si-model model=gdn simulator.d_model=16,32,64` trains a Gaussian Density Network on data simulated from a (homogeneous) Susceptible-Infected model, sweeping over three different network widths.

### Implemented Simulators

#### Simulation Experiments
- Normal/Normal conjugate model (for testing purposes)
- Bayesian Linear Regression model (for testing purposes)
- Susceptible-Infected (SI) transmission model (homogeneous or heterogeneous transmission rates, complete or partial observation of cases)

#### Empirical Models
- SI model for carbapenem-resistant Klebsiella pneumonia (CRKP) transmission; requires confidential data

### Implemented Posterior Estimators
- Gaussian Density Network
- Normalizing Flow (RealNVP)
- Approximate Bayesian Computation
