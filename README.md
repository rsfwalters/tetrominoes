# Tetrominoes dataset

This repository contains the Tetrominoes dataset, used to assess the generalization performance of variational autodencoders. 

If you use this dataset in your work, please cite it as follows:

## Bibtex

```
@misc{tetrominoes19,
author = {Alican Bozkurt and Babak Esmaeili and Jennifer Dy and Dana Brooks and Jan-Willem van de Meent},
title = {Tetrominoes dataset},
howpublished= {https://github.com/neu-pml/tetrominoes/},
year = "2019",
}
```

## Description

Tetrominoes is a dataset of 2D shapes procedurally generated from 6 ground truth
independent latent factors. These factors are *rotation*, *color*, *scale*, *x* and *y* positions, and *shape*.

## Generating the Tetromino dataset

To generate and save the Tetromino dataset, run:

```
python generate_tetromino_h5.py --fname data/train-tetro.h5 --num_timesteps 10000 --seed 1; python generate_tetromino_h5.py --fname data/test-tetro.h5 --num_timesteps 10000 --seed 2
```
