from typing import List, Dict
import pprint
import numpy
import os
import pickle
import random
import json
import sys

from optimal_model.Experiment import Experiment

if __name__ == "__main__":
    
    if len(sys.argv) != 2 or sys.argv[1] not in ["small_plentiful", "small_scarce"]:
        print("Usage python3 run_optimization_model.py <scenario>.")
        print("Scenarios: small_plentiful, small_scarce")
        sys.exit(1)
 
    # File locations
    opt_model_json_file = f"config/{sys.argv[1]}/opt_model.json"
    slice_definitions_json_file = f"config/{sys.argv[1]}/slices.json"
    seeds_json_file = "config/seeds.json"
    
    # Seeds used for randomness
    with open(seeds_json_file, "r") as f:
        seeds = json.load(f)

    # Reading slice definitions
    with open(slice_definitions_json_file, "r") as f:
        slice_definitions = json.load(f)
    
    # Reading configurations for the optimization model
    with open(opt_model_json_file, "r") as f:
        opt_model_configs = json.load(f)

    # Extracting slice requirements
    slice_requirements = {}
    for slice_name, configs in slice_definitions.items():
        slice_requirements[configs["id"]] = {}
        for req_name in configs["requirements"].keys():
            if req_name == "long_term_capacity":
                new_req_name = "ltc"
            elif req_name == "capacity":
                new_req_name = "cap"
            elif req_name == "latency":
                new_req_name = "lat"
            else:
                raise ValueError(f"Unknown requirement name: {req_name}")
            
            slice_requirements[configs["id"]][new_req_name] = {
                "req": configs["requirements"][req_name],
                "weight": configs["requirement_weights"][req_name]
            }
    
    # Extracting other slice configurations
    slice_demands = {
        configs["id"]: configs["user_config"]["flow_throughput"]
        for configs in slice_definitions.values()
    }
    slice_pkt_size = {
        configs["id"]: configs["user_config"]["pkt_size"]
        for configs in slice_definitions.values()
    }
    slice_buffer_size = { # In packets
        configs["id"]: configs["user_config"]["buffer_size"]/slice_pkt_size[configs["id"]]
        for configs in slice_definitions.values()
    }
    slice_weights = {
        configs["id"]: configs["weight"]
        for configs in slice_definitions.values()
    }
    slice_max_latencies = { # In TTIs
        configs["id"]: configs["user_config"]["max_lat"]
        for configs in slice_definitions.values()
    }
    slice_intra_schedulers = {
        configs["id"]: configs["intra_scheduler"]
        for configs in slice_definitions.values()
    }

    # Scenario
    scenarios = [
        (
            configs["n_rbgs"], # Number of RBGs
            configs["n_ttis"], # Number of TTIs
            False, # has PER
            configs["slice_drift_aggregation_method"], # Aggregation_method
            configs["minimize_resources"], # Changes the objective function if resource minimization is a goal
            { # Number of UEs per slice
                slice_definitions[slice_name]["id"]: n_ues
                for slice_name, n_ues in configs["slice_users"].items()
                if n_ues > 0
            },
            slice_intra_schedulers, # Intra-slice scheduling for each slice
            seed,
            configs["time_limit"]
            
        )
        for configs in opt_model_configs.values()
        for seed in seeds
    ]

    # Preparing the experiments
    exp = Experiment(
        workers=16,
        slice_requirements=slice_requirements,
        slice_demands=slice_demands,
        slice_pkt_size=slice_pkt_size,
        slice_buffer_size=slice_buffer_size,
        slice_weights=slice_weights,
        slice_max_latencies=slice_max_latencies,
        coding_gain=5.0, # dB
    )

    # Preparing environment for saving the results
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists(f"./results/{sys.argv[1]}"):
        os.makedirs(f"./results/{sys.argv[1]}")

    # Executing the experiment for each scenario
    for seed in seeds:
        for opt_name, configs in opt_model_configs.items():
            model_name, solution, results = exp.execute_experiment(
                seed=seed,
                n_rbgs=configs["n_rbgs"],
                rbg_size=configs["rbg_size"],
                n_ttis=configs["n_ttis"],
                has_per=False,
                aggregation_method=configs["slice_drift_aggregation_method"],
                slice_n_ues={ # Number of UEs per slice
                    slice_definitions[slice_name]["id"]: n_ues
                    for slice_name, n_ues in configs["slice_users"].items()
                    if n_ues > 0
                },
                slice_intra_schedulings=slice_intra_schedulers,
                resource_minimization=configs["minimize_resources"],
                time_limit=configs["time_limit"],
                tti_length=configs["tti_length"], # TTI length in secods
                rb_bandwidth=configs["rb_bandwidth"], # Bandwidth of 1 RB in Hz
                window=configs["metrics_window"], # Window size (in TTIs) for the ltc metric and pf intra-slice scheduler
                rbg_grouping_method=configs["rbg_grouping_method"], # "random_block" or "big_block"
            )
            print("Solution: {}".format(solution))

            if results is None:
                print("No optimal/feasible solution found for model {}.".format(model_name))
                continue
            
            # Saving the results as a pickle file
            results_file = f"./results/{sys.argv[1]}/{model_name}.pickle"
            with open(results_file, "wb") as file:
                pickle.dump(results, file)
            print(f"Results saved as {results_file}")

            # Printing the results
            pprint.pp(results)
            n_rbgs = len(results["R"])
            S = results["S"]
            print ("Ratio of allocated resources per step:{}".format([sum(results['rho'][u,r,t] for r in range(n_rbgs) for s in S for u in results["U"][s])/n_rbgs for t in range(configs["n_ttis"])]))
            print("RBGs allocated for each slice per step:")
            for s in S:
                print("Slice {}: {}".format(s,[sum(results['rho'][u,r,t] for r in range(n_rbgs) for u in results["U"][s]) for t in range(configs["n_ttis"])]))
            print("Allocated RBGs:")
            for t in range(configs["n_ttis"]):
                print(f"Step {t}")
                for s in S:
                    for u in results["U"][s]:
                        print(f"UE {u} (Slice {s}): {[r for r in range(n_rbgs) if results['rho'][u,r,t] == 1]}")


# Access pickle files with:
# with open('filename.pickle', 'rb') as file:
#    data = pickle.load(file)

# Solution object methods:
# get_solve_status() - Returns the status of the solution (optimal, infeasible, unbounded, or unknown)
# get_objective_value() - Returns the value of the objective function
# get_solve_time() - Returns the time (in seconds) spent to solve the model
# get_stop_cause() - Returns the reason why the optimization stopped (time_limit, optimal_solution, or unknown)