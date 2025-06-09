# DREAMIN: Channel-Aware Inter-Slices Radio Resource Scheduling for Efficient SLA Assurance

[Paper (no link until publication)](LINK_TO_THE_PAPER)

[ICC presentation slides](https://github.com/LABORA-INF-UFG/paper-DGMK-2024/blob/main/icc_presentation.pdf)

The Drift and Resource Allocation Minimization (DREAMIN) scheduler is a channel-aware inter-slice radio resource scheduler oriented to efficiently reducing SLA drift and resource usage. This repository has all the code we implemented to execute the experiments we show in our paper. It contains:
- `simulation/` - our simulation of a downlink 5G scenario
- `simulation/intersched.py` - implementations of inter-slice scheduling algorithms, including DREAMIN, [RadioSaber](https://www.usenix.org/conference/nsdi23/presentation/chen-yongzhou), and a Weighted Round-Robin
- `simulation/intrasched.py` - implementations of intra-slice scheduling algorithms, including Round-Robin, Proportional Fairness, and Maximum Throughput
- `optimal_model/` - implementation of our problem formulation in [docplex](https://pypi.org/project/docplex/) to run on IBM's Constraint Programming Optimizer 
- `cqi-traces-noise0/` - traces dataset from [RadioSaber](https://www.usenix.org/conference/nsdi23/presentation/chen-yongzhou), originally available [here](https://github.com/elvinlife/RadioSaber/blob/main/cqi-traces-noise0/)
- `config/` - CQI mapping tables, seed values for reproductible randomness, slice requirements, user demands, and other parameters for each experimental scenario
- `run_optimization_model.py` - script for solving the formulated problem using IBM's Constraint Programming Optimizer and saving its results in `results/` 
- `main.py` - script for running the simulation and saving its metrics in `metrics/`
- `generate_plots.py` - script for reading simulation metrics in `metrics/` and generating plots in `plots/`
- `execute_experiments.sh` - script that automates all experiments and saves outputs in `outputs/`

# How to cite us

Use the bibtex below (fields with ? will be filled after publication):

```bibtex
@inproceedings{icc_dgmk_2024,
        author = {Daniel Campos and Gabriel M. Almeida and Mohammad J. Abdel-Rahman and Kleber V. Cardoso},
        title = {DREAMIN: Channel-Aware Inter-Slices Radio Resource Scheduling for Efficient SLA Assurance},
        booktitle = {ICC 2025-IEEE International Conference on Communications},
        location = {Montreal, Canada},
        year = {2025},
        keywords = {Service level agreement, network slicing, resource block, radio resource scheduling, energy efficiency},
        organization={IEEE},
        issn = {?},
        pages = {?},
        doi = {?},
        url = {?}
        
}
```

# Reproducing the experiments

All experiments were executed in a Thinkpad E14 gen 4 (40 GB RAM and Intel i7-1255U 12 cores 4.7 GHz) running Ubuntu 20.04.

## Setup the environment

As all code is written in Python, this is the only language you need to have installed.
You'll also need IBM's Contraint Programming Optimizer to execute the optimization model.
To build the same environment we used in our evaluations, install:
- Python 3.8.10 with pip 20.0.2
- IBM ILOG CPLEX Optimization Studio 22.1.0 

**Python dependencies:**
- tqdm
- numpy
- typing
- matplotlib
- pandas
- scipy
- docplex

Install all Python dependencies by running:

```bash
pip3 install tqdm numpy typing matplotlib pandas scipy docplex
```

Additionally, make sure the `cpooptimizer_bin_path` parameter at line 330 from `optimal_model/Experiment.py` matches the path to the binaries of IBM's Constraint Programming Optimizer.

## Executing the experiments

In this work, we evaluate 4 different scenarios, each one with 3 slices:
- `small_plentiful` - 10 TTIs, 1 user per slice, and enough resources to meet all SLA requirements for all users
- `small_scarce` - 10 TTIs, 1 user per slice, and higher SLA requirements that cannot be met for every user at every TTI
- `large_plentiful` - 475 TTIs, 1 user per slice, and enough resources to meet all SLA requirements for all users during most of the time
- `large_scarce` - 475 TTIs, 3 user per slice, and not enough resources to meet all SLA requirements for all users

Due to the limitations of optimally solving the implemented optimization model, we can only execute the solver for small scenarios.

Each scenario is executed 20 times.
At each time, a different set of users is selected, defined by the seed in `config/seeds.json`.
Note that the solver, which has a time limit of 1 hour to find the best solution, will solve 40 different problems (20 times the `small_plentiful` + 20 times the `small_scarce`), totalizing 40 hours of execution.
You can change the `time_limit` fields in `config/small_plentiful/opt_model.json` and `config/small_scarce/opt_model.json` to adjust how many seconds one search will take.

After obtaining the solutions from the optimization model, you'll be able to run the simulations.
Each simulation executes 20 times, following the same sets of users defined by `config/seeds.json`.
A different simulation is executed for each one of the evaluated inter-slice schedulers.
You can change which inter-slice schedulers will be evaluated in `config/small_plentiful/simulation.json`, `config/small_scarce/simulation.json`, `config/large_plentiful/simulation.json`, and `config/large_scarce/simulation.json`.

### The `execute_experiments.sh` script

To run the optimization model and simulations and generate the plots with only one command, you may execute:

```bash
bash execute_experiments.sh
```

At the end, you'll be able to check:
- The collected metrics from the simulations, saved as .csv files inside `metrics/`
- The generated plots inside `plots/`
- The outputs from the executed commands in `outputs/`

If you want to execute only certain steps or scenarios of the experiment, check the following subsections.

### Running the optimization model

To run the optimization model, execute:

```bash
python3 run_optimization_model.py <scenario> 
```

where `<scenario>` may be `small_plentiful` or `small_scarce`.

### Running the simulations

To run the simulations, execute:

```bash
python3 main.py <scenario>
```

where `<scenario>` may be `small_plentiful`, `small_scarce`, `large_plentiful`, or `large_scarce`.

Note that, as the simulation config files for `small_plentiful` and `small_scarce` scenarios include the `Optimal` inter-slice scheduler, you need to have the results from the optimization model stored in `results/` to execute this step. 

### Generating plots

To generate the plots, execute:

```bash
python3 generate_plots.py <scenario>
```

where `<scenario>` may be `small_plentiful`, `small_scarce`, `large_plentiful`, or `large_scarce`.

Note that you need to execute the simulations of a scenario before generating its plots so the collected metrics are stored in `metrics/` for analysis. 

# Contact

If you have any questions, please email Daniel "Dante" Campos: danielcampossilva@inf.ufg.br
