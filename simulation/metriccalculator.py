from typing import Dict
import pandas as pd

from .simulation import Simulation

class UserCalculator:
    def __init__(self, sim: Simulation):
        self.sim = sim
    
    def get_n_allocated_rbgs(self, user_id:int) -> int:
        return len(self.sim.users[user_id].rbgs)

    def get_capacity(self, user_id:int) -> float:
        return sum(self.sim.ue_tti_rbg_cap[user_id][self.sim.step][r] for r in self.sim.users[user_id].rbgs)
    
    def get_channel_unaware_ue_cap(self, user_id:int) -> float:
        return sum(self.sim.ue_tti_rbg_cap[user_id][self.sim.step][r] for r in self.sim.rbgs)/len(self.sim.rbgs)
    
    def get_occ_buff_bits(self, user_id:int) -> int:
        return sum(self.sim.users[user_id].buffer.buff)*self.sim.users[user_id].buffer.pkt_size
    
    def get_latency(self, user_id:int) -> float:
        return self.sim.users[user_id].buffer.oldest_pkt_lat
    
    # TODO
    def get_pkt_error_rate(self, user_id:int) -> float:
        return 0 
    
    def get_arrival_throughput(self, user_id:int) -> float:
        return (
            self.sim.users[user_id].buffer.arriv_pkts
            *self.sim.users[user_id].buffer.pkt_size
            /self.sim.TTI
        )

    def get_sent_throughput(self, user_id:int) -> float:
        return (
            self.sim.users[user_id].buffer.sent_pkts
            *self.sim.users[user_id].buffer.pkt_size
            /self.sim.TTI
        )
    
    def get_long_term_throughput(self, user_id:int, step_window:int, df: pd.DataFrame) -> float:
        actual_window = min(step_window, self.sim.step+1)
        total_pkts = (
            sum(df[(df["step"] < self.sim.step) & (df["step"] > self.sim.step-actual_window)]["sent_pkts"])
            +self.sim.users[user_id].buffer.sent_pkts
        )
        return (
            total_pkts
            *self.sim.users[user_id].buffer.pkt_size
            /(self.sim.TTI*actual_window)
        )
    
    def get_long_term_capacity(self, user_id:int, step_window:int, df: pd.DataFrame) -> float:
        actual_window = min(step_window, self.sim.step+1)
        sum_cap = (
            sum(df[(df["step"] < self.sim.step) & (df["step"] > self.sim.step-actual_window)]["capacity"])
            +self.get_capacity(user_id)
        )
        return sum_cap/actual_window
    
    def get_arriv_pkts(self, user_id:int) -> int:
        return self.sim.users[user_id].buffer.arriv_pkts
    
    def get_sent_pkts(self, user_id:int) -> int:
        return self.sim.users[user_id].buffer.sent_pkts
    
    def get_dropp_pkts_buff_full(self, user_id:int) -> int:
        return self.sim.users[user_id].buffer.dropped_pkts_buffer_full

    def get_dropp_pkts_max_lat(self, user_id:int) -> int:
        return self.sim.users[user_id].buffer.dropped_pkts_max_lat
    
    def get_dropp_pkts_total(self, user_id:int) -> int:
        return self.sim.users[user_id].buffer.dropped_pkts_buffer_full + self.sim.users[user_id].buffer.dropped_pkts_max_lat
    
    def get_pkts_in_buffer(self, user_id:int) -> int:
        return sum(self.sim.users[user_id].buffer.buff)
    
    def get_capacity_drift(self, user_id:int) -> float:
        if "capacity" not in self.sim.users[user_id].requirements:
            return 0
        cap = self.get_capacity(user_id)
        req = self.sim.users[user_id].requirements["capacity"]
        return 0 if cap >= req else (req - cap)/req
    
    def get_long_term_capacity_drift(self, user_id:int, step_window:int, df: pd.DataFrame) -> float:
        if "long_term_capacity" not in self.sim.users[user_id].requirements:
            return 0
        ltc = self.get_long_term_capacity(user_id, step_window, df)
        req = self.sim.users[user_id].requirements["long_term_capacity"]
        return 0 if ltc >= req else (req - ltc)/req
    
    def get_latency_drift(self, user_id:int) -> float:
        if "latency" not in self.sim.users[user_id].requirements:
            return 0
        lat = self.get_latency(user_id)
        req = self.sim.users[user_id].requirements["latency"]
        max_lat = self.sim.users[user_id].buffer.max_lat
        return 0 if lat <= req else (lat - req)/(max_lat-req)
    
    def get_drift(self, user_id:int, step_window:int, df: pd.DataFrame) -> float:
        drift = 0
        if "capacity" in self.sim.users[user_id].requirements:
            drift += self.get_capacity_drift(user_id) * self.sim.users[user_id].requirement_weights["capacity"]
        if "long_term_capacity" in self.sim.users[user_id].requirements:
            drift += self.get_long_term_capacity_drift(user_id, step_window, df) * self.sim.users[user_id].requirement_weights["long_term_capacity"]
        if "latency" in self.sim.users[user_id].requirements:
            drift += self.get_latency_drift(user_id) * self.sim.users[user_id].requirement_weights["latency"]
        return drift

class SliceCalculator:
    def __init__(self, sim: Simulation, user_calc: UserCalculator):
        self.sim = sim
        self.user_calc = user_calc
    
    def get_n_allocated_rbgs(self, slice_id:int) -> int:
        return sum(
            len(u.rbgs)
            for u in self.sim.slices[slice_id].users.values()
        )

    def get_worst_ue_capacity(self, slice_id:int) -> float:
        return min(
            sum(self.sim.ue_tti_rbg_cap[u.id][self.sim.step][r] for r in u.rbgs)
            for u in self.sim.slices[slice_id].users.values()
        )
    
    def get_best_ue_capacity(self, slice_id:int) -> float:
        return max(
            sum(self.sim.ue_tti_rbg_cap[u.id][self.sim.step][r] for r in u.rbgs)
            for u in self.sim.slices[slice_id].users.values()
        )
    
    def get_total_capacity(self, slice_id:int) -> float:
        return sum(
            sum(self.sim.ue_tti_rbg_cap[u.id][self.sim.step][r] for r in u.rbgs)
            for u in self.sim.slices[slice_id].users.values()
        )

    def get_avg_capacity(self, slice_id:int) -> float:
        return sum(
            sum(self.sim.ue_tti_rbg_cap[u.id][self.sim.step][r] for r in u.rbgs)
            for u in self.sim.slices[slice_id].users.values()
        )/len(self.sim.slices[slice_id].users)

    def get_avg_channel_unaware_ue_cap(self, slice_id:int) -> float:
        return sum(
            sum(self.sim.ue_tti_rbg_cap[u.id][self.sim.step][r] for r in self.sim.rbgs)/len(self.sim.rbgs)
            for u in self.sim.slices[slice_id].users.values()
        )/len(self.sim.slices[slice_id].users)
     
    def get_avg_occ_buff_bits(self, slice_id:int) -> float:
        return sum(
            sum(u.buffer.buff)*u.buffer.pkt_size
            for u in self.sim.slices[slice_id].users.values()
        )/len(self.sim.slices[slice_id].users)
    
    def get_avg_latency(self, slice_id:int) -> float:
        total_latency = 0.0
        for u in self.sim.slices[slice_id].users.values():
            total_latency += self.user_calc.get_latency(u.id)
        return total_latency/len(self.sim.slices[slice_id].users)
    
    def get_avg_pkt_error_rate(self, slice_id:int) -> float:
        return 0 # TODO
    
    def get_total_arrival_throughput(self, slice_id:int) -> float:
        return sum(
            u.buffer.arriv_pkts*u.buffer.pkt_size/self.sim.TTI
            for u in self.sim.slices[slice_id].users.values()
        )
    
    def get_total_sent_throughput(self, slice_id:int) -> float:
        return sum(
            u.buffer.sent_pkts*u.buffer.pkt_size/self.sim.TTI
            for u in self.sim.slices[slice_id].users.values()
        )
    
    def get_avg_long_term_throughput(self, slice_id:int, step_window:int, users_df: Dict[int,pd.DataFrame]) -> float:
        actual_window = min(step_window, self.sim.step+1)
        total_sent_bits = sum(
            (
                sum(users_df[u.id][(users_df[u.id]["step"] < self.sim.step) & (users_df[u.id]["step"] > self.sim.step-actual_window)]["sent_pkts"])
                +self.sim.users[u.id].buffer.sent_pkts 
            )
            *self.sim.users[u.id].buffer.pkt_size
            for u in self.sim.slices[slice_id].users.values()
        )
        return total_sent_bits/(self.sim.TTI*actual_window)

    def get_avg_long_term_capacity(self, slice_id:int, step_window:int, users_df: Dict[int,pd.DataFrame]) -> float:
        actual_window = min(step_window, self.sim.step+1)
        total_cap = sum(
            (
                sum(users_df[u.id][(users_df[u.id]["step"] < self.sim.step) & (users_df[u.id]["step"] > self.sim.step-actual_window)]["capacity"])
                +sum(self.sim.ue_tti_rbg_cap[u.id][self.sim.step][r] for r in self.sim.users[u.id].rbgs)
            )
            for u in self.sim.slices[slice_id].users.values()
        )
        return total_cap/actual_window
    
    def get_total_arriv_pkts(self, slice_id:int) -> int:
        return sum(
            u.buffer.arriv_pkts
            for u in self.sim.slices[slice_id].users.values()
        )
    
    def get_total_sent_pkts(self, slice_id:int) -> int:
        return sum(
            u.buffer.sent_pkts
            for u in self.sim.slices[slice_id].users.values()
        )
    
    def get_total_pkts_in_buffer(self, slice_id:int) -> int:
        return sum(
            sum(u.buffer.buff)
            for u in self.sim.slices[slice_id].users.values()
        )

    def get_total_dropp_pkts_buff_full(self, slice_id:int) -> int:
        return sum(
            u.buffer.dropped_pkts_buffer_full
            for u in self.sim.slices[slice_id].users.values()
        )
    
    def get_total_dropp_pkts_max_lat(self, slice_id:int) -> int:
        return sum(
            u.buffer.dropped_pkts_max_lat
            for u in self.sim.slices[slice_id].users.values()
        )
    
    def get_total_dropp_pkts_total(self, slice_id:int) -> int:
        return sum(
            u.buffer.dropped_pkts_buffer_full + u.buffer.dropped_pkts_max_lat
            for u in self.sim.slices[slice_id].users.values()
        )

    def get_drift(self, slice_id:int, step_window:int, users_df: Dict[int,pd.DataFrame]) -> float:
        return sum(self.user_calc.get_drift(u.id, step_window, users_df[u.id]) for u in self.sim.slices[slice_id].users.values())/len(self.sim.slices[slice_id].users)
    
    def get_capacity_fairness(self, slice_id:int) -> float:
        capacities = [self.user_calc.get_capacity(u.id) for u in self.sim.slices[slice_id].users.values()]
        if sum(capacities) == 0:
            return 1.0
        jain_fairness = sum(capacities)**2/(len(capacities)*sum(c**2 for c in capacities))
        return jain_fairness

    def get_drift_fairness(self, slice_id:int, step_window:int, users_df: Dict[int,pd.DataFrame]) -> float:
        drifts = [self.user_calc.get_drift(u.id, step_window, users_df[u.id]) for u in self.sim.slices[slice_id].users.values()]
        if sum(drifts) == 0:
            return 1.0
        jain_fairness = sum(drifts)**2/(len(drifts)*sum(c**2 for c in drifts))
        return jain_fairness
    
    def get_resource_usage(self, slice_id:int) -> float:
        return len(self.sim.slices[slice_id].rbgs)/len(self.sim.rbgs)

class SimulationCalculator:
    def __init__(self, sim: Simulation, slice_calc: SliceCalculator):
        self.sim = sim
        self.slice_calc = slice_calc

    def get_total_rbg_allocation(self,) -> int:
        return sum(
            len(u.rbgs)
            for u in self.sim.users.values()
        )

    def get_total_capacity(self) -> float:
        return sum(self.slice_calc.get_total_capacity(s.id) for s in self.sim.slices.values())
    
    def get_scheduler_time(self,) -> float:
        return self.sim.scheduler_time
    
    def get_drift(self, step_window:int, users_df: Dict[int,pd.DataFrame]) -> float:
        return sum(self.slice_calc.get_drift(s.id, step_window, users_df)*s.weight for s in self.sim.slices.values())

    def get_objective(self, step_window:int, users_df: Dict[int,pd.DataFrame]) -> float:
        drift = self.get_drift(step_window, users_df)
        if drift > 0:
            return 1 + drift
        else:
            return self.get_total_rbg_allocation()/len(self.sim.rbgs)
    
    def get_resource_usage(self,) -> float:
        return self.get_total_rbg_allocation()/len(self.sim.rbgs)
    
    def get_capacity_fairness(self) -> float:
        return sum(self.slice_calc.get_capacity_fairness(s.id)*s.weight for s in self.sim.slices.values())

    def get_drift_fairness(self, step_window:int, users_df: Dict[int,pd.DataFrame]) -> float:
        return sum(self.slice_calc.get_drift_fairness(s.id, step_window, users_df)*s.weight for s in self.sim.slices.values())
    
    def get_inter_slice_fairness(self, step_window:int, users_df: Dict[int,pd.DataFrame]) -> float:
        drifts = [self.slice_calc.get_drift(s.id, step_window, users_df)*s.weight for s in self.sim.slices.values()]
        if sum(drifts) == 0:
            return 1.0
        jain_fairness = sum(drifts)**2/(len(drifts)*sum(c**2 for c in drifts))
        return jain_fairness