import json
import sys
import numpy as np

from simulation.plotmanager import PlotManager

if __name__ == "__main__":

    if len(sys.argv) != 2 or sys.argv[1] not in ["small_plentiful", "small_scarce", "large_plentiful", "large_scarce"]:
        print("Usage python3 new_plots.py <scenario>.")
        print("Scenarios: small_plentiful, small_scarce, large_plentiful, large_scarce")
        sys.exit(1)

    metrics_folder = f"metrics/{sys.argv[1]}"
    plots_folder = f"plots/{sys.argv[1]}"
    simulation_json_file = f"config/{sys.argv[1]}/simulation.json"
    slice_json_file = f"config/{sys.argv[1]}/slices.json"

    # Seeds
    with open("config/seeds.json", "r") as f:
        seeds = json.load(f)

    # Simulations
    with open(simulation_json_file, "r") as f:
        simulation = json.load(f)
    sim_names = simulation.keys()

    # Slices
    with open(slice_json_file, "r") as f:
        slices = json.load(f)
    slice_names = {config["id"]: config["type"] for config in slices.values()}

    # Plotting colors
    colors = {
        "RS": "orange",
        "DREAMIN": "red",
        "Optimal": "blue",
        "WRR": "green",
        "Non-GBR": "purple",
        "GBR": "pink",
        "DC-GBR": "cyan",
    }

    # Renaming labels only on plots
    rename = {
        "Optimal": "APPR"
    }

    # Instatiating the plot manager
    plotter = PlotManager(
        sim_names=sim_names,
        slice_names=slice_names,
        colors=colors,
        seeds=seeds,
        metrics_folder=metrics_folder,
        plots_folder=plots_folder,
        rename=rename
    )
    
    # # Plots for each experiment
    # for seed in seeds:
        
    #     # Plots for each simulation
    #     for sim_name in sim_names:
            
    #         # Comparing slice metrics for the same simulation
    #         plotter.plot_metric_line_one_seed_one_sim_multi_slices(sim_name, seed=seed, metric="n_allocated_rbgs")
    #         plotter.plot_metric_line_one_seed_one_sim_multi_slices(sim_name, seed=seed, metric="slice_drift")
    #         plotter.plot_metric_line_one_seed_one_sim_multi_slices(sim_name, seed=seed, metric="total_capacity")
    #         plotter.plot_metric_line_one_seed_one_sim_multi_slices(sim_name, seed=seed, metric="worst_ue_capacity")
    #         plotter.plot_metric_line_one_seed_one_sim_multi_slices(sim_name, seed=seed, metric="best_ue_capacity")
    #         plotter.plot_metric_line_one_seed_one_sim_multi_slices(sim_name, seed=seed, metric="avg_capacity")
    #         plotter.plot_metric_line_one_seed_one_sim_multi_slices(sim_name, seed=seed, metric="avg_lat")
    #         plotter.plot_metric_line_one_seed_one_sim_multi_slices(sim_name, seed=seed, metric="capacity_fairness")
    #         plotter.plot_metric_line_one_seed_one_sim_multi_slices(sim_name, seed=seed, metric="drift_fairness")

    #         # Comparing user metrics for the same simulation
    #         plotter.plot_metric_line_one_seed_one_sim_multi_users(sim_name, seed=seed, metric="n_allocated_rbgs")
    #         plotter.plot_metric_line_one_seed_one_sim_multi_users(sim_name, seed=seed, metric="drift")
    #         plotter.plot_metric_line_one_seed_one_sim_multi_users(sim_name, seed=seed, metric="capacity")
    #         plotter.plot_metric_line_one_seed_one_sim_multi_users(sim_name, seed=seed, metric="in_buffer_pkts")
    #         plotter.plot_metric_line_one_seed_one_sim_multi_users(sim_name, seed=seed, metric="cap_drift")
    #         plotter.plot_metric_line_one_seed_one_sim_multi_users(sim_name, seed=seed, metric="ltc_drift")
    #         plotter.plot_metric_line_one_seed_one_sim_multi_users(sim_name, seed=seed, metric="lat_drift")

    #     # Plots for each slice
    #     for slice_id in slice_names.keys():
            
    #         # Comparing different simulations for the same slice
    #         plotter.plot_metric_line_one_seed_one_slice_multi_sim(sim_names, seed=seed, slice_id=slice_id, metric="n_allocated_rbgs")
    #         plotter.plot_metric_line_one_seed_one_slice_multi_sim(sim_names, seed=seed, slice_id=slice_id, metric="slice_drift")
    #         plotter.plot_metric_line_one_seed_one_slice_multi_sim(sim_names, seed=seed, slice_id=slice_id, metric="total_capacity")
    #         plotter.plot_metric_line_one_seed_one_slice_multi_sim(sim_names, seed=seed, slice_id=slice_id, metric="worst_ue_capacity")
    #         plotter.plot_metric_line_one_seed_one_slice_multi_sim(sim_names, seed=seed, slice_id=slice_id, metric="best_ue_capacity")
    #         plotter.plot_metric_line_one_seed_one_slice_multi_sim(sim_names, seed=seed, slice_id=slice_id, metric="avg_capacity")
    #         plotter.plot_metric_line_one_seed_one_slice_multi_sim(sim_names, seed=seed, slice_id=slice_id, metric="avg_lat")
    #         plotter.plot_metric_line_one_seed_one_slice_multi_sim(sim_names, seed=seed, slice_id=slice_id, metric="capacity_fairness")
    #         plotter.plot_metric_line_one_seed_one_slice_multi_sim(sim_names, seed=seed, slice_id=slice_id, metric="drift_fairness")
        
    # # Plots comparing different simulations
    # for seed in seeds:
        
    #     # Bar plots
    #     plotter.plot_metric_bar_one_seed(sim_names, seed=seed, metric="drift")
    #     plotter.plot_metric_bar_one_seed(sim_names, seed=seed, metric="resource_usage")
    #     plotter.plot_metric_bar_one_seed(sim_names, seed=seed, metric="total_capacity")
    #     plotter.plot_metric_bar_one_seed(sim_names, seed=seed, metric="objective")
    #     plotter.plot_metric_bar_one_seed(sim_names, seed=seed, metric="capacity_fairness")
    #     plotter.plot_metric_bar_one_seed(sim_names, seed=seed, metric="drift_fairness")
  
    #     # Line plots
    #     plotter.plot_metric_line_one_seed(sim_names, seed=seed, metric="drift")
    #     plotter.plot_metric_line_one_seed(sim_names, seed=seed, metric="resource_usage")
    #     plotter.plot_metric_line_one_seed(sim_names, seed=seed, metric="total_capacity")
    #     plotter.plot_metric_line_one_seed(sim_names, seed=seed, metric="objective")
    #     plotter.plot_metric_line_one_seed(sim_names, seed=seed, metric="capacity_fairness")
    #     plotter.plot_metric_line_one_seed(sim_names, seed=seed, metric="drift_fairness")

    # # Plots aggregating all seeds

    # if "small" in sys.argv[1]:
    #     figsize=(8,3.5)
    #     fontsize=18
    # else:
    #     figsize=(8,6)
    #     fontsize=20

    # # CDFs
    # plotter.plot_cdf_all_seeds(sim_names=sim_names, seeds=seeds, metric="drift", xlabel="Drift",figsize=figsize, fontsize=fontsize)
    # plotter.plot_cdf_all_seeds(sim_names=sim_names, seeds=seeds, metric="resource_usage", xlabel="Resource usage", x_as_percentage=True, figsize=figsize, fontsize=fontsize)
    # plotter.plot_cdf_all_seeds(sim_names=sim_names, seeds=seeds, metric="total_capacity", xlabel="Total capacity (Mbps)", figsize=figsize, fontsize=fontsize)
    # plotter.plot_cdf_all_seeds(sim_names=sim_names, seeds=seeds, metric="objective", xlabel="Objective value", figsize=figsize, fontsize=fontsize)
    # plotter.plot_cdf_all_seeds(sim_names=sim_names, seeds=seeds, metric="capacity_fairness", xlabel="Fairness", x_as_percentage=True, figsize=figsize, fontsize=fontsize)
    # plotter.plot_cdf_all_seeds(sim_names=sim_names, seeds=seeds, metric="drift_fairness", xlabel="Drift fairness", x_as_percentage=True, figsize=figsize, fontsize=fontsize)
    # plotter.plot_cdf_all_seeds(sim_names=sim_names, seeds=seeds, metric="inter_slice_fairness", xlabel="Inter-slice fairness", x_as_percentage=True, figsize=figsize, fontsize=fontsize)

    # if "small" in sys.argv[1]:
    #     figsize=(6,4)
    #     fontsize=22
    # else:
    #     figsize=(8,6)
    #     fontsize=20

    # # Bar plots
    # plotter.plot_metric_bar_all_seeds(sim_names=sim_names, seeds=seeds, metric="drift", ylabel="Drift", figsize=figsize, fontsize=fontsize)
    # plotter.plot_metric_bar_all_seeds(sim_names=sim_names, seeds=seeds, metric="resource_usage", ylabel="Resource usage", y_as_percentage=True, figsize=figsize, fontsize=fontsize)
    # plotter.plot_metric_bar_all_seeds(sim_names=sim_names, seeds=seeds, metric="total_capacity", ylabel="Total capacity (Mbps)", figsize=figsize, fontsize=fontsize)
    # plotter.plot_metric_bar_all_seeds(sim_names=sim_names, seeds=seeds, metric="objective", ylabel="Objective value", figsize=figsize, fontsize=fontsize)
    # plotter.plot_metric_bar_all_seeds(sim_names=sim_names, seeds=seeds, metric="capacity_fairness", ylabel="Capacity fairness", y_as_percentage=True, figsize=figsize, fontsize=fontsize)
    # plotter.plot_metric_bar_all_seeds(sim_names=sim_names, seeds=seeds, metric="drift_fairness", ylabel="Drift fairness", y_as_percentage=True, figsize=figsize, fontsize=fontsize)
    # plotter.plot_metric_bar_all_seeds(sim_names=sim_names, seeds=seeds, metric="inter_slice_fairness", ylabel="Inter-slice fairness", y_as_percentage=True, figsize=figsize, fontsize=fontsize)

    # # Boxplots
    # plotter.plot_metric_boxplot_all_seeds(sim_names=sim_names, seeds=seeds, metric="drift", ylabel="Drift", figsize=figsize, fontsize=fontsize)
    # plotter.plot_metric_boxplot_all_seeds(sim_names=sim_names, seeds=seeds, metric="resource_usage", ylabel="Resource usage", y_as_percentage=True, figsize=figsize, fontsize=fontsize)
    # plotter.plot_metric_boxplot_all_seeds(sim_names=sim_names, seeds=seeds, metric="total_capacity", ylabel="Total capacity (Mbps)", figsize=figsize, fontsize=fontsize)
    # plotter.plot_metric_boxplot_all_seeds(sim_names=sim_names, seeds=seeds, metric="objective", ylabel="Objective value", figsize=figsize, fontsize=fontsize)
    # plotter.plot_metric_boxplot_all_seeds(sim_names=sim_names, seeds=seeds, metric="capacity_fairness", ylabel="Capacity fairness", y_as_percentage=True, figsize=figsize, fontsize=fontsize)
    # plotter.plot_metric_boxplot_all_seeds(sim_names=sim_names, seeds=seeds, metric="drift_fairness", ylabel="Drift fairness", y_as_percentage=True, figsize=figsize, fontsize=fontsize)
    # plotter.plot_metric_boxplot_all_seeds(sim_names=sim_names, seeds=seeds, metric="inter_slice_fairness", ylabel="Inter-slice fairness", y_as_percentage=True, figsize=figsize, fontsize=fontsize)

    if "small_plentiful" in sys.argv[1]:
        figsize=(4,4)
        fontsize=17
        plotter.plot_metric_bar_all_seeds(sim_names=["RS", "WRR"], seeds=seeds, metric="drift", ylabel="SLAd", figsize=figsize, fontsize=fontsize)
        plotter.plot_metric_bar_all_seeds(sim_names=["Optimal", "DREAMIN"], seeds=seeds, metric="resource_usage", ylabel="Resource usage", y_as_percentage=True, figsize=figsize, fontsize=fontsize)
        plotter.plot_metric_boxplot_all_seeds(sim_names=["Optimal", "DREAMIN","RS", "WRR"], seeds=seeds, metric="drift", ylabel="SLAd", figsize=figsize, fontsize=fontsize, rotate_x_ticks=True)
        plotter.plot_metric_boxplot_all_seeds(sim_names=["Optimal", "DREAMIN","RS", "WRR"], seeds=seeds, metric="resource_usage", ylabel="Resource usage", y_as_percentage=True, figsize=figsize, fontsize=fontsize, rotate_x_ticks=True, yticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
        plotter.plot_metric_bar_all_seeds_one_slice(sim_names=["RS", "WRR", "Optimal", "DREAMIN"], seeds=seeds, metric="slice_drift", slice_id=0, ylabel="SLAd", figsize=figsize, fontsize=fontsize)
        plotter.plot_metric_bar_all_seeds_one_slice(sim_names=["RS", "WRR","Optimal", "DREAMIN"], seeds=seeds, metric="resource_usage", slice_id=0, ylabel="Resource usage", y_as_percentage=True, figsize=figsize, fontsize=fontsize)
        plotter.plot_metric_bar_all_seeds_one_slice_min(sim_names=["RS", "WRR","Optimal", "DREAMIN"], seeds=seeds, metric="resource_usage", slice_id=0, ylabel="Resource usage", y_as_percentage=True, figsize=figsize, fontsize=fontsize)
        plotter.plot_metric_bar_all_seeds_one_slice_max(sim_names=["RS", "WRR","Optimal", "DREAMIN"], seeds=seeds, metric="resource_usage", slice_id=0, ylabel="Resource usage", y_as_percentage=True, figsize=figsize, fontsize=fontsize)

    if "small_scarce" in sys.argv[1]:
        figsize=(8,3.5)
        fontsize=18
        plotter.plot_cdf_all_seeds(sim_names=sim_names, seeds=seeds, metric="drift", xlabel="SLAd",figsize=figsize, fontsize=fontsize, yticks=[0, 0.25, 0.5, 0.75, 1])

    if "large_plentiful" in sys.argv[1]:
        figsize=(4,4)
        fontsize=17
        plotter.plot_metric_bar_all_seeds(sim_names=["RS", "WRR", "DREAMIN"], seeds=seeds, metric="drift", ylabel="SLAd", figsize=figsize, fontsize=fontsize)
        plotter.plot_metric_bar_all_seeds(sim_names=["DREAMIN", "RS", "WRR"], seeds=seeds, metric="resource_usage", ylabel="Resource usage", y_as_percentage=True, figsize=figsize, fontsize=fontsize)
        plotter.plot_metric_boxplot_all_seeds(sim_names=["DREAMIN", "RS", "WRR"], seeds=seeds, metric="drift", ylabel="SLAd", figsize=figsize, fontsize=fontsize)
        plotter.plot_metric_boxplot_all_seeds(sim_names=["DREAMIN", "RS", "WRR"], seeds=seeds, metric="resource_usage", ylabel="Resource usage", figsize=figsize, fontsize=fontsize, y_as_percentage=True,yticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
        plotter.plot_metric_bar_all_seeds_one_slice(sim_names=["RS", "WRR", "DREAMIN"], seeds=seeds, metric="slice_drift", slice_id=0, ylabel="SLAd", figsize=figsize, fontsize=fontsize)
        plotter.plot_metric_bar_all_seeds_one_slice(sim_names=["RS", "WRR", "DREAMIN"], seeds=seeds, metric="resource_usage", slice_id=0, ylabel="Resource usage", y_as_percentage=True, figsize=figsize, fontsize=fontsize)
        plotter.plot_metric_bar_all_seeds_one_slice_min(sim_names=["RS", "WRR", "DREAMIN"], seeds=seeds, metric="resource_usage", slice_id=0, ylabel="Resource usage", y_as_percentage=True, figsize=figsize, fontsize=fontsize)
        plotter.plot_metric_bar_all_seeds_one_slice_max(sim_names=["RS", "WRR", "DREAMIN"], seeds=seeds, metric="resource_usage", slice_id=0, ylabel="Resource usage", y_as_percentage=True, figsize=figsize, fontsize=fontsize)

    if "large_scarce" in sys.argv[1]:
        figsize=(7,2)
        fontsize=14
        plotter.plot_cdf_all_seeds(sim_names=sim_names, seeds=seeds, metric="drift", xlabel="SLAd",figsize=figsize, fontsize=fontsize,yticks=[0, 0.25, 0.5, 0.75, 1])
        plotter.plot_cdf_all_seeds(sim_names=sim_names, seeds=seeds, metric="capacity_fairness", xlabel="Intra-slice fairness", figsize=figsize, fontsize=fontsize, yticks=[0, 0.25, 0.5, 0.75, 1], ccdf=True)
        plotter.plot_metric_bar_all_seeds(sim_names=sim_names, seeds=seeds, metric="drift", ylabel="SLAd", figsize=figsize, fontsize=fontsize)

        # Getting how many TTIs have at least a minimum value of fairness
        for sim in sim_names:
            data = []
            for seed in seeds:
                data.extend(list(plotter.simulation_data[seed][sim]["capacity_fairness"]))
            data = np.array(data)
            minimum_fairness = 0.9
            ttis_satisfied = np.sum(data > minimum_fairness)
            print(f"Simulation {sim}: {ttis_satisfied/len(data) * 100:.1f}% of TTIs have at least 0.9 fairness")