# Graph-based Time Series Clustering for End-to-End Hierarchical Forecasting (ICML 2024)

[![ICML](https://img.shields.io/badge/ICML-2024-blue.svg?style=flat-square)](https://openreview.net/forum?id=nd47Za5jk5)
[![PDF](https://img.shields.io/badge/%E2%87%A9-PDF-orange.svg?style=flat-square)](https://openreview.net/pdf?id=nd47Za5jk5)
[![arXiv](https://img.shields.io/badge/arXiv-2305.19183-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2305.19183)

This repository contains the code for the reproducibility of the controlled experiments presented in the paper "Graph-based Time Series Clustering for End-to-End Hierarchical Forecasting" (ICML 2024).

**Authors**: [Andrea Cini](mailto:andrea.cini@usi.ch), Danilo Mandic, Cesare Alippi

---

## In a nutshell

HiGP (Hierarchical Graph Predictor) is a graph-based methodology unifying relational and hierarchical inductive biases in the context of deep learning for time series forecasting.

---

## Directory structure

The directory is structured as follows:

```
.
├── config/
├── lib/
├── tsl/
├── conda_env.yaml
├── default_config.yaml
└── experiments/
    └── run_benchmark.py

```

## Datasets

The datasets used in the experiments are provided by the `tsl` library. The CER-E dataset can be obtained for research purposes following the instructions at this [link](https://www.ucd.ie/issda/data/commissionforenergyregulationcer/).

## Configuration files

The `config` directory stores the configuration files used to run the experiments.

## Requirements

To solve all dependencies, we recommend using Anaconda and the provided environment configuration by running the command:

```bash
conda env create -f conda_env.yml
conda activate higp
```

A custom version of the `tsl` library is included in the folder, we suggest to set the PYTHONPATH accordingly.

## Experiments

The script used for the experiments in the paper is in the `experiments` folder.

* `run_benchmark.py` is used to train and evaluate models on the datasets considered in the study. As an example, to run the HiGP-TTS model on the METR-LA dataset:

	```
	python -m experiments.run_benchmark config=default model=higp_tts dataset=la 
	```
 
## Bibtex reference

If you find this code useful please consider citing our paper:

```
@article{cini2024graph,
title        = {{Graph-based Time Series Clustering for End-to-End Hierarchical Forecasting}},
author       = {Cini, Andrea and Mandic, Danilo and Alippi, Cesare},
journal      = {International Conference on Machine Learning},
year         = 2024
}
```