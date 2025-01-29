from typing import List, Dict, Tuple
import pandas as pd
import json

from .simulation import Simulation
from .metriccalculator import SimulationCalculator, SliceCalculator, UserCalculator

class MetricManager:
    def __init__(self, sim: Simulation, step_window: int) -> None:
        self.sim = sim
        self.step_window = step_window
        self.sim_metrics:pd.DataFrame = pd.DataFrame()
        self.slice_metrics:Dict[int, pd.DataFrame] = {s: pd.DataFrame() for s in sim.slices.keys()}
        self.user_metrics:Dict[int, pd.DataFrame] = {u: pd.DataFrame(columns=["step", "capacity", "sent_pkts"]) for u in sim.users.keys()}
        self.user_calculator = UserCalculator(sim)
        self.slice_calculator = SliceCalculator(sim, self.user_calculator)
        self.sim_calculator = SimulationCalculator(sim, self.slice_calculator)
  
    
    def collect_simulation_metrics(self) -> None:   
        new_row = pd.DataFrame([{
            "step": self.sim.step,
            "total_capacity": self.sim_calculator.get_total_capacity(),
            "n_allocated_rbgs": self.sim_calculator.get_total_rbg_allocation(),
            "resource_usage": self.sim_calculator.get_resource_usage(),
            "scheduling_time": self.sim_calculator.get_scheduler_time(),
            "drift": self.sim_calculator.get_drift(self.step_window, self.user_metrics),
            "objective": self.sim_calculator.get_objective(self.step_window, self.user_metrics),
            "capacity_fairness": self.sim_calculator.get_capacity_fairness(),
            "drift_fairness": self.sim_calculator.get_drift_fairness(self.step_window, self.user_metrics),
            "inter_slice_fairness": self.sim_calculator.get_inter_slice_fairness(self.step_window, self.user_metrics),
        }])
        self.sim_metrics = pd.concat([self.sim_metrics, new_row], ignore_index=True, copy=False)

    def collect_slice_metrics(self, slice_id:int) -> None:
        new_row = pd.DataFrame([{
            "step": self.sim.step,
            "n_allocated_rbgs": self.slice_calculator.get_n_allocated_rbgs(slice_id),
            "resource_usage": self.slice_calculator.get_resource_usage(slice_id),
            "total_capacity": self.slice_calculator.get_total_capacity(slice_id),
            "worst_ue_capacity": self.slice_calculator.get_worst_ue_capacity(slice_id),
            "best_ue_capacity": self.slice_calculator.get_best_ue_capacity(slice_id),
            "avg_capacity": self.slice_calculator.get_avg_capacity(slice_id),
            "avg_channel_unaware_cap": self.slice_calculator.get_avg_channel_unaware_ue_cap(slice_id),
            "avg_occ_buff_bits": self.slice_calculator.get_avg_occ_buff_bits(slice_id),
            "avg_lat": self.slice_calculator.get_avg_latency(slice_id),
            "avg_pkt_error_rate": self.slice_calculator.get_avg_pkt_error_rate(slice_id),
            "total_arriv_thr": self.slice_calculator.get_total_arrival_throughput(slice_id),
            "total_sent_thr": self.slice_calculator.get_total_sent_throughput(slice_id),
            "avg_long_term_thr": self.slice_calculator.get_avg_long_term_throughput(slice_id, self.step_window, self.user_metrics),
            "avg_long_term_cap": self.slice_calculator.get_avg_long_term_capacity(slice_id, self.step_window, self.user_metrics),
            "total_arriv_pkts": self.slice_calculator.get_total_arriv_pkts(slice_id),
            "total_sent_pkts": self.slice_calculator.get_total_sent_pkts(slice_id),
            "total_in_buffer_pkts": self.slice_calculator.get_total_pkts_in_buffer(slice_id),
            "total_dropp_pkts_buff_full": self.slice_calculator.get_total_dropp_pkts_buff_full(slice_id),
            "total_dropp_pkts_max_lat": self.slice_calculator.get_total_dropp_pkts_max_lat(slice_id),
            "total_dropp_pkts_total": self.slice_calculator.get_total_dropp_pkts_total(slice_id),
            "slice_drift": self.slice_calculator.get_drift(slice_id, self.step_window, self.user_metrics),
            "capacity_fairness": self.slice_calculator.get_capacity_fairness(slice_id),
            "drift_fairness": self.slice_calculator.get_drift_fairness(slice_id, self.step_window, self.user_metrics),
        }])
        self.slice_metrics[slice_id] = pd.concat([self.slice_metrics[slice_id], new_row], ignore_index=True, copy=False)
    
    def collect_user_metrics(self, user_id:int) -> None:
        new_row = pd.DataFrame([{
            "step": self.sim.step,
            "n_allocated_rbgs": self.user_calculator.get_n_allocated_rbgs(user_id),
            "capacity": self.user_calculator.get_capacity(user_id),
            "channel_unaware_cap": self.user_calculator.get_channel_unaware_ue_cap(user_id),
            "occ_buff_bits": self.user_calculator.get_occ_buff_bits(user_id),
            "lat": self.user_calculator.get_latency(user_id),
            "pkt_error_rate": self.user_calculator.get_pkt_error_rate(user_id),
            "arriv_thr": self.user_calculator.get_arrival_throughput(user_id),
            "sent_thr": self.user_calculator.get_sent_throughput(user_id),
            "long_term_thr": self.user_calculator.get_long_term_throughput(user_id, self.step_window, self.user_metrics[user_id]),
            "long_term_cap": self.user_calculator.get_long_term_capacity(user_id, self.step_window, self.user_metrics[user_id]),
            "arriv_pkts": self.user_calculator.get_arriv_pkts(user_id),
            "sent_pkts": self.user_calculator.get_sent_pkts(user_id),
            "in_buffer_pkts": self.user_calculator.get_pkts_in_buffer(user_id),
            "dropp_pkts_buff_full": self.user_calculator.get_dropp_pkts_buff_full(user_id),
            "dropp_pkts_max_lat": self.user_calculator.get_dropp_pkts_max_lat(user_id),
            "dropp_pkts_total": self.user_calculator.get_dropp_pkts_total(user_id),
            "cap_drift": self.user_calculator.get_capacity_drift(user_id),
            "ltc_drift": self.user_calculator.get_long_term_capacity_drift(user_id, self.step_window, self.user_metrics[user_id]),
            "lat_drift": self.user_calculator.get_latency_drift(user_id),
            "drift": self.user_calculator.get_drift(user_id, self.step_window, self.user_metrics[user_id]),
        }])
        self.user_metrics[user_id] = pd.concat([self.user_metrics[user_id], new_row], ignore_index=True, copy=False)
    
    def collect_metrics(self,) -> None:
        self.collect_simulation_metrics()
        for slice_id in self.sim.slices.keys():
            self.collect_slice_metrics(slice_id)
        for user_id in self.sim.users.keys():
            self.collect_user_metrics(user_id)
    
    def save_metrics(self, folder:str) -> None:
        self.sim_metrics.to_csv(f"{folder}/sim_metrics.csv", index=False)
        for slice_id, slice_metrics in self.slice_metrics.items():
            slice_metrics.to_csv(f"{folder}/slice_{slice_id}_metrics.csv", index=False)
        for user_id, user_metrics in self.user_metrics.items():
            user_metrics.to_csv(f"{folder}/user_{user_id}_metrics.csv", index=False)
        with open(f"{folder}/description.json", "w") as f:
            json.dump({
                "slices": {s.id: {
                    "type": s.type,
                    "requirements": s.requirements,
                    "users": list(s.users.keys()),
                } for s in self.sim.slices.values()},
            }, f, indent=4)