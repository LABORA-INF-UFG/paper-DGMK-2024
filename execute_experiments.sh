#! /bin/bash

# Creating a folder for the outputs
mkdir -p outputs

# Running the optimization model
python3 run_optimization_model.py small_scarce > outputs/output_opt_model_small_scarce.txt
python3 run_optimization_model.py small_plentiful > outputs/output_opt_model_small_plentiful.txt

# Running the simulation
python3 main.py small_scarce > outputs/output_simulation_small_scarce.txt
python3 main.py small_plentiful > outputs/output_simulation_small_plentiful.txt
python3 main.py large_scarce > outputs/output_simulation_large_scarce.txt
python3 main.py large_plentiful > outputs/output_simulation_large_plentiful.txt

# Generating the plots
python3 generate_plots.py small_scarce > outputs/output_plots_small_scarce.txt
python3 generate_plots.py small_plentiful > outputs/output_plots_small_plentiful.txt
python3 generate_plots.py large_scarce > outputs/output_plots_large_scarce.txt
python3 generate_plots.py large_plentiful > outputs/output_plots_large_plentiful.txt