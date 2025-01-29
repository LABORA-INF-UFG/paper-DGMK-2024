import numpy as np
from tqdm import tqdm
from typing import List, Dict
import os
import json
import time
import random
import sys

from simulation.simulation import Simulation
from simulation.metricmanager import MetricManager
from simulation import intersched, intrasched

def generate_ue_tti_rbg_cap(
    ue_ids:List[int],
    rb_bandwidth:float,
    rbg_size:int,
    n_rbgs:int,
    n_ttis:int,
    method:str="big_block", # "big_block" or "random_block"
) -> Dict[int, Dict[int, Dict[int, float]]]:
    
    # Reading CQI traces from files
    ue_cqi_path = "./cqi-traces-noise0/ue{}.log"
    ue_cqis = dict()
    for u in ue_ids:
        with open(ue_cqi_path.format(u), "r") as f:
            ue_cqis[u] = []
            for line in f:
                row = list(map(int, line.strip().split()))
                ue_cqis[u].append(row)
    
    # Reading the CQI to spectral efficiency mapping
    with open("config/cqi_to_spec_eff.json", "r") as f:
        data = json.load(f)
        cqi_to_spec_eff = {int(key): value for key, value in data.items()}

    # Reading the CQI to code rate mapping
    with open("config/cqi_to_code_rate.json", "r") as f:
        data = json.load(f)
        cqi_to_code_rate = {int(key): value for key, value in data.items()}
    
    # Shuffling RBGs if using random block method
    if method == "random_block":
        rbgs = [i*4 for i in range(int(512/4))]
        random.shuffle(rbgs)
        rbgs = rbgs[:n_rbgs]

    # Calculating the capacity for each UE-TTI-RBG combination
    ue_tti_rbg_cap = dict()
    for u in ue_ids:
        ue_tti_rbg_cap[u] = dict()
        for t in range(n_ttis):
            ue_tti_rbg_cap[u][t] = dict()
            for r in range(n_rbgs):
                if method == "random_block":
                    ue_tti_rbg_cap[u][t][r] = 4 * rb_bandwidth * cqi_to_spec_eff[ue_cqis[u][t][rbgs[r]]] * cqi_to_code_rate[ue_cqis[u][t][rbgs[r]]]
                elif method == "big_block":
                    ue_tti_rbg_cap[u][t][r] = sum(rb_bandwidth * cqi_to_spec_eff[ue_cqis[u][t][r*rbg_size+i]] * cqi_to_code_rate[ue_cqis[u][t][r*rbg_size+i]] for i in range(rbg_size)) # Average capacity for the RBG
    
    return ue_tti_rbg_cap

if __name__ == "__main__":
    
    if len(sys.argv) != 2 or sys.argv[1] not in ["small_plentiful", "small_scarce", "large_plentiful", "large_scarce"]:
        print("Usage python3 main.py <scenario>.")
        print("Scenarios: small_plentiful, small_scarce, large_plentiful, large_scarce")
        sys.exit(1)

    # Configuration files locations
    simulation_json_file = f"config/{sys.argv[1]}/simulation.json"
    slice_definitions_json_file = f"config/{sys.argv[1]}/slices.json"
    seeds_json_file = "config/seeds.json"

    # Reading seeds used for randomness
    with open(seeds_json_file, "r") as f:
        seeds = json.load(f)
    
    # Reading simulation configurations and slice definitions
    with open(simulation_json_file, "r") as f:
        sim_configs = json.load(f)
    with open(slice_definitions_json_file, "r") as f:
        slice_definitions = json.load(f)

    # Instantiating and running each simulation
    for seed in seeds:
        for sim_name in sim_configs:
            # Setting the random seed for reproducibility
            np.random.seed(seed)
            random.seed(seed)

            # Selecting users for the simulation 
            ue_ids = list(range(158)) # There are 158 UEs in the dataset
            np.random.shuffle(ue_ids) # Randomizing users (randomness defined by np seed)
            total_n_users=sum([n_users for n_users in sim_configs[sim_name]["slice_users"].values()])
            ue_ids = ue_ids[:total_n_users] # Selecting the first total_n_users users

            # Reading CQI values and converting them to capacities
            sub_carrier_width = 2**sim_configs[sim_name]["numerology"] * 15e3 # Hz
            ue_tti_rbg_cap = generate_ue_tti_rbg_cap(
                ue_ids=ue_ids,
                rb_bandwidth= 12 * sub_carrier_width,
                rbg_size=sim_configs[sim_name]["rbg_size"],
                n_rbgs=sim_configs[sim_name]["n_rbgs"],
                n_ttis=sim_configs[sim_name]["n_ttis"],
                method=sim_configs[sim_name]["rbg_grouping_method"],
            )

            # Instantiating the simulation scheduler
            if sim_configs[sim_name]["scheduler"]["name"] == "round_robin":
                inter_scheduler = intersched.RoundRobin()
            elif sim_configs[sim_name]["scheduler"]["name"] == "weighted_round_robin":
                inter_scheduler = intersched.WeightedRoundRobin()
            elif sim_configs[sim_name]["scheduler"]["name"] == "omniscient_heuristic":
                inter_scheduler = intersched.OmniscientHeuristic(
                    window=sim_configs[sim_name]["scheduler"]["window"],
                    metrics_window=sim_configs[sim_name]["step_window"],
                    n_ttis=sim_configs[sim_name]["n_ttis"],
                    step=0,
                    tti_lenght=2**-sim_configs[sim_name]["numerology"] * 1e-3
                )
            elif sim_configs[sim_name]["scheduler"]["name"] == "step_drift_heuristic":
                inter_scheduler = intersched.StepwiseDriftHeuristic(
                    metrics_window=sim_configs[sim_name]["step_window"],
                    step=0,
                )
            elif sim_configs[sim_name]["scheduler"]["name"] == "optimal":
                inter_scheduler = intersched.OptimalScheduler(
                    scenario=sys.argv[1],
                    n_ttis=sim_configs[sim_name]["n_ttis"],
                    rbg_size=sim_configs[sim_name]["rbg_size"],
                    n_rbgs=sim_configs[sim_name]["n_rbgs"],
                    n_ues=sum(ues for ues in sim_configs[sim_name]["slice_users"].values()),
                    n_slices=sum(1 if ues > 0 else 0 for ues in sim_configs[sim_name]["slice_users"].values()),
                    time_limit=sim_configs[sim_name]["scheduler"]["time_limit"],
                    seed=seed,
                    step = 0
                )
            elif sim_configs[sim_name]["scheduler"]["name"] == "radiosaber":
                inter_scheduler = intersched.RadiosaberScheduler(step = 0)
            elif sim_configs[sim_name]["scheduler"]["name"] == "modified_radiosaber":
                inter_scheduler = intersched.ModifiedRadiosaberScheduler(
                    metrics_folder=f"metrics/{sys.argv[1]}",
                    sim_name=sim_configs[sim_name]["scheduler"]["sim_name"],
                    seed=seed,
                    step = 0
                )
            else:
                raise ValueError("Invalid inter-scheduler")
            
            # Instantiating the simulation
            sim = Simulation(
                experiment_name=sim_name,
                scheduler=inter_scheduler,
                numerology=sim_configs[sim_name]["numerology"], # TTI = 1ms
                rbgs = list(range(sim_configs[sim_name]["n_rbgs"])),
                rng = np.random.default_rng(seed=seed),
                ue_tti_rbg_cap=ue_tti_rbg_cap,
            )

            # Instantiating slices and users
            user_index = 0
            for slice_name, slice_n_users in sim_configs[sim_name]["slice_users"].items():
                
                # Instantiating the intra-slice scheduler for each slice
                if slice_definitions[slice_name]["intra_scheduler"] == "round_robin":
                    intra_scheduler = intrasched.RoundRobin()
                elif slice_definitions[slice_name]["intra_scheduler"] == "maximum_throughput":
                    intra_scheduler = intrasched.MaximumThroughput(ue_tti_rbg_cap)
                elif slice_definitions[slice_name]["intra_scheduler"] == "proportional_fair":
                    intra_scheduler = intrasched.ProportionalFair(ue_tti_rbg_cap, window=sim_configs[sim_name]["step_window"])
                else:
                    raise ValueError("Invalid intra-scheduler")
                            
                if slice_n_users > 0:
                    sim.add_slice(
                        slice_id=slice_definitions[slice_name]["id"],
                        slice_type=slice_definitions[slice_name]["type"],
                        requirements=slice_definitions[slice_name]["requirements"],
                        requirement_weights=slice_definitions[slice_name]["requirement_weights"],
                        weight=slice_definitions[slice_name]["weight"],
                        # weight=( # Weight is the proportion of the slice's total demand
                        #     (slice_definitions[slice_name]["user_config"]["flow_throughput"] * slice_n_users)
                        #     /sum(slice_definitions[name]["user_config"]["flow_throughput"] * n for name, n in sim_configs[sim_name]["slice_users"].items())
                        # ),
                        intra_scheduler=intra_scheduler,
                    )
                    sim.add_users(
                        slice_id=slice_definitions[slice_name]["id"],
                        user_ids=ue_ids[user_index:user_index+slice_n_users],
                        max_lat=slice_definitions[slice_name]["user_config"]["max_lat"],
                        buffer_size=slice_definitions[slice_name]["user_config"]["buffer_size"],
                        pkt_size=slice_definitions[slice_name]["user_config"]["pkt_size"],
                        flow_type=slice_definitions[slice_name]["user_config"]["flow_type"],
                        flow_throughput=slice_definitions[slice_name]["user_config"]["flow_throughput"],
                    )
                    user_index += slice_n_users
            
            # Extra steps for the omniscient heuristic scheduler (pre-calculates all allocations)
            if isinstance(sim.scheduler, intersched.OmniscientHeuristic):
                arrived_pkts = sim.generate_arrived_pkts(n_ttis=sim_configs[sim_name]["n_ttis"])
                t = time.time()
                sim.scheduler.calculate_allocation(ue_tti_rbg_cap=ue_tti_rbg_cap, arrived_pkts=arrived_pkts, slices=sim.slices, rbgs=sim.rbgs)
                print(sim.scheduler.allocation)
                print("Time to calculate allocation:", time.time()-t, "seconds")

            # Instantiating the metric manager
            metric_manager = MetricManager(sim, sim_configs[sim_name]["step_window"])

            # Running all TTIs
            for _ in tqdm(range(sim_configs[sim_name]["n_ttis"]), leave=False, desc="TTIs"):
                sim.arrive_packets()
                sim.schedule_rbgs()
                sim.transmit()
                metric_manager.collect_metrics()
                sim.advance_step()
            
            # Saving the metrics in a file
            folder = f"metrics/{sys.argv[1]}/{sim_name}_{seed}"
            if not os.path.exists(folder): # Create folder if it doesn't exist
                os.makedirs(folder)
            metric_manager.save_metrics(folder=folder)
            
            print(f"Finished simulation {sim_name}_{seed}")