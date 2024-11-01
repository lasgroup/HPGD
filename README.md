# Contextual Bilevel Reinforcement Learning
[<img src="https://img.shields.io/badge/license-Apache2.0-blue.svg">](https://github.com/luchris429/purejaxrl/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

![](figure.png)

This code repository accompanies the paper "Contextual Bilevel Reinforcement Learning"
by [Vinzenz Thoma](https://vinzenzthoma.com/), [Barna Pásztor](https://pasztorb.github.io), [Andreas Krause](https://las.inf.ethz.ch/krausea),
[Giorgia Ramponi](https://gioramponi.github.io/), and [Yifan Hu](https://sites.google.com/view/yifan-hu).
The work is accepted at the Neural Information Processing Systems (NeurIPS) 2024 conference.
Preprint available on [arXiv](https://arxiv.org/abs/2406.01575).

### Requirements

Use Python 3.10 or above and to install requirements in a clean environment, run the following command:

```setup
pip install -r requirements.txt
```
We recommend using a virtual environment, e.g., Conda, to avoid conflicts with other packages.


# Four-Rooms Environment

To train HPGD, run this command:
```train
python train_stochastic_bilevel_opt.py --experiment_dir <path_to_data>
```
We provided the `config.yaml` files required to reproduce the results reported in the paper in the `data/experiment_reg_lambda_0_00*` directories.

## Comparison algorithms
To train the algorithm (AMD) from [Chen et al. 2022](https://proceedings.mlr.press/v162/chen22ab.html), run the following command:
```train
python train_amd.py --experiment_dir <path_to_data>
```
and to train the Zero-Order gradient estimator, run the following command:
```train
python train_zero_order.py --experiment_dir <path_to_data>
```

## Evaluation and visualization
The evaluation and visualization scripts are provided in the `notebooks/experiment_visualization.ipynb` notebook.
Run this notebook to reproduce the Figures and Tables from the paper.


## Results

Our model achieves the following performance on (Table 1) from Section 4 of the paper:

| MDP Reg. Parameter | Upper-Level Regularization parameter | HPGD             | AMD              | Zero-Order        |
|--------------------|--------------------------------------|------------------|------------------|-------------------|
| 0.001              | 1                                    | **0.91** ± 0.088 | 0.58 ± 0.000     | 0.59 ± 0.059      |
| 0.001              | 3                                    | 0.51 ± 0.006     | 0.51 ± 0.000     | 0.50 ± 0.005      |
| 0.001              | 5                                    | 0.46 ± 0.006     | 0.46 ± 0.003     | 0.46 ± 0.007      |
| 0.003              | 1                                    | 0.95 ± 0.002     | 1.00 ± 0.000     | 0.91 ± 0.048      |
| 0.003              | 3                                    | **0.73** ± 0.001 | 0.39 ± 0.000     | 0.40 ± 0.028      |
| 0.003              | 5                                    | 0.29 ± 0.003     | 0.32 ± 0.000     | 0.32 ± 0.002      |
| 0.005              | 1                                    | 1.17 ± 0.011     | 1.28 ± 0.003     | 1.15 ± 0.026      |
| 0.005              | 3                                    | 1.01 ± 0.002     | 1.13 ± 0.004     | 1.02 ± 0.027      |
| 0.005              | 5                                    | 0.87 ± 0.003     | 0.97 ± 0.009     | 0.79 ± 0.027      |

Performance over hyperparameters for the Four-Rooms Problem averaged over 10 random seeds with standard errors.
Algorithms perform on-par for most hyperparameters while HPGD outperforms others in few.
AMD enjoys low variance due to the non-stochastic gradient updates while Zero-Order suffers from the most variation.


# Tax Design Environment

To train HPGD, run this command:
```train
python train_tax_design.py --experiment_dir <path_to_data>
```
and to train the Zero-Order gradient estimator, run the following command:
```train
python train_tax_design_zero_order.py --experiment_dir <path_to_data>
```
We provided the `config.yaml` files required to reproduce the results reported in the paper in the `data/experiment_tax_design_reg_lambda_0_00*` directories.

## Evaluation and visualization
The evaluation and visualization scripts are provided in the `notebooks/experiment_visualization_tax_design.ipynb` notebook.
Run this notebook to reproduce the Figures and Tables from the paper.

# References and Contact
With any question about the code, please reach out to [Barna Pásztor](mailto:barna.pasztor@ai.ethz.ch) and,
if you find our code useful for your research, cite our work as follows:
```bibtex
@misc{thoma2024stochasticbileveloptimizationlowerlevel,
      title={Stochastic Bilevel Optimization with Lower-Level Contextual Markov Decision Processes}, 
      author={Vinzenz Thoma and Barna Pasztor and Andreas Krause and Giorgia Ramponi and Yifan Hu},
      year={2024},
      eprint={2406.01575},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2406.01575}, 
}
```