[![DOI](https://zenodo.org/badge/1103544770.svg)](https://doi.org/10.5281/zenodo.17712808)

# Active learning with physics-informed neural networks for optimal sensor placement in deep tunneling through transversely isotropic elastic rocks

This repository contains code and data to reproduce the results of the paper 

>**Active learning with physics-informed neural networks for optimal sensor placement in deep tunneling through transversely isotropic elastic rocks**, by *Alec Tristani* and *Chloé Arson*

In this study, we explore a sequential active learning strategy to best position field extensometer and convergence sensors to both reconstruct the displacement field and learn the rock mass constitutive parameters for deep circular tunnels excavated in transversely isotropic elastic rocks.


## Requirements

This code requires `pytorch` available at https://pytorch.org/
  

## Contents

- `examples/` contains the code and the results for both extensometer and convergence modes

- `module/pool_utils.py` defines the training pool and dataset

- `module/model.py` implements the architecture of the physics-informed neural network model

- `module/train.py` implements the training functions

- `module/losses.py` implements the loss functions of the model based on a transversely isotropic elastic constitutive law

- `module/select_sensors.py` implements the selection of sensors based on Monte Carlo Dropout

- `module/active_learning.py` implements the active training of the model

- `module/visualization.py` contains various functions for visualizing the results

- `moose/` contains the Moose and mesh codes to generate the raw data

- `synthetic_data/` contains the raw data

## Usage

- `convergence_mode.ipynb` and `extensometer_mode.ipynb` contain the main active learning loop for adding convergence and extensometer measurements, respectively.  
- Each notebook provides an illustrative example of the sequential training.

<!-- ## Code citation

If you found this code useful, please consider citing it:
```
@misc{tristani2026activelearningtunnelcode,
  author       = {Tristani, Alec},
  title        = {Active learning with physics-informed neural networks for optimal sensor placement in deep tunneling through transversely isotropic elastic rocks},
  year         = 2026,
  howpublished = {Code available at GitHub},
  url          = {https://github.com/Alec-YT/Active-Learning-Tunnel},
}
``` -->
