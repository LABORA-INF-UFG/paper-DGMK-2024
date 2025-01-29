from abc import ABC, abstractmethod
from typing import Dict, Tuple, List
import copy
import pickle
import random
import pandas as pd

from simulation.slice import Slice
from simulation.user import User
import simulation.intrasched as intrasched

class InterSliceScheduler(ABC):
    @abstractmethod
    def schedule(self, slices: Dict[int, Slice], users: Dict[int, User], rbgs: List[int]):
        raise Exception("Called abstract InterSliceScheduler method")

    @abstractmethod
    def advance_step(self) -> None:
        raise Exception("Called abstract IntraSliceScheduler method")

class RoundRobin(InterSliceScheduler):
    """
    Round Robin scheduler that cycles through slices distributing one RBG at a time.
    """
    
    def __init__(self, offset: int = 0) -> None:
        self.offset = offset

    def schedule(self, slices: Dict[int, Slice], users: Dict[int, User], rbgs: List[int]):
        ids = [s.id for s in slices.values()]
        self.offset %= len(ids)

        # Randomizing the RBGs
        random_rbgs = copy.copy(rbgs)
        random.shuffle(random_rbgs)

        for r in random_rbgs:
            slices[ids[self.offset]].allocate_rbg(r)
            self.offset = (self.offset + 1) % len(ids)
    
    def advance_step(self) -> None:
        pass

class WeightedRoundRobin(InterSliceScheduler):
    """
    Round Robin scheduler that distributes random RBGs in the proportion of the slices' weights.
    """
    
    def __init__(self, step=0) -> None:
        self.rbs_offset = None
        self.step = step

    # Calculates the number of RBGs to allocate to each slice
    def slice_quota(self, slices: Dict[int, Slice], n_rbgs:int) -> Dict[int, int]:
        
        # Initializing data structures
        if self.rbs_offset is None:
            self.rbs_offset = {s.id: 0.0 for s in slices.values()}
        rbs_share = {}
        rbgs_quota = {}
        
        # Calculating how many RBGs each slice should receive
        for s in slices.values():
            rbs_share[s.id] = n_rbgs*s.weight + self.rbs_offset[s.id] # The offset includes the previous fractional part
            rbgs_quota[s.id] = int(rbs_share[s.id]) # Flooring the number of RBGs
        
        # Randomly distributing the remaining RBGs
        extra_rbgs = n_rbgs - sum(rbgs_quota.values())
        while extra_rbgs > 0:
            random_slice = random.choice(list(slices.keys()))
            rbgs_quota[random_slice] += 1
            extra_rbgs -= 1
        
        # Calculating the new offsets
        for s in slices.values():
            self.rbs_offset[s.id] = rbs_share[s.id] - rbgs_quota[s.id]
        
        return rbgs_quota

    def schedule(self, slices: Dict[int, Slice], users: Dict[int, User], rbgs: List[int]):
        
        # Shuffling RBGs
        random_rbgs = copy.copy(rbgs)
        random.shuffle(random_rbgs)

        # Getting the number of RBGs per slice
        quotas = self.slice_quota(slices, len(rbgs))

        # Allocating the random RBGs
        for s in slices.values():
            for _ in range(quotas[s.id]):
                s.allocate_rbg(random_rbgs.pop(0))
    
    def advance_step(self) -> None:
        self.step += 1

class DummyScheduler(InterSliceScheduler):
    """
    Dummy scheduler that receives an allocation and schedules it.
    """
    
    def __init__(self,) -> None:
        self.allocation:Dict[int, List[int]] = None # allocation[slice_id] = rbg_ids
    
    def set_allocation(self, allocation: Dict[int, List[int]]) -> None:
        self.allocation: Dict[int, int] = allocation

    def schedule(self, slices: Dict[int, Slice], users: Dict[int, User], rbgs: List[int]) -> None:
        for s in slices.values():
            for rbg in self.allocation[s]:
                s.allocate_rbg(rbg)
    
    def advance_step(self) -> None:
        pass

class OmniscientHeuristic(InterSliceScheduler):
    """
    Heuristic whose schedules rely on knowing the future channel conditions and user demands.
    """
    def __init__(
        self,
        window: int, # Number of steps to look ahead
        metrics_window:int, # Time window (in TTIs) for long term capacity and proportional fair score 
        n_ttis:int,
        step:int = 0,
        tti_lenght:float = 1e-3,
    ) -> None:
        self.window = window
        self.metrics_window = metrics_window
        self.step = step
        self.n_ttis = n_ttis
        self.tti_lenght = tti_lenght
        self.allocation:Dict[int, Dict[int, List[int]]] = None # allocation[step][slice_id] = rbg_ids
        self.slices:Dict[int, Slice] = None
    
    def create_user_data(self):
        """
        Creates user data structures considering the requirements of the slices.
        Considers a null allocation (the capacity for all users is 0).
        """
        
        # Initializes the user data structure
        self.user_data = {u.id: {} for s in self.slices.values() for u in s.users.values()}
        for s in self.slices.values():
            
            # Common data for all users
            for u in s.users.values():
                self.user_data[u.id]["capacity"] = [0.0]*self.n_ttis
                self.user_data[u.id]["drift"] = [0.0]*self.n_ttis
                self.user_data[u.id]["total_drift"] = [0.0]*self.n_ttis
            
            # Data for users of slices with Proportional Fair scheduler
            if isinstance(s.scheduler, intrasched.ProportionalFair): 
                for u in s.users.values():
                    self.user_data[u.id]["historical_capacity"] = [s.scheduler.starting_historical_capacity*(1.0 - 1.0/self.metrics_window)**t for t in range(self.n_ttis)]
            
            # Data for users of slices with capacity requirements
            if s.requirements.get("capacity") is not None: 
                for u in s.users.values():
                    self.user_data[u.id]["capacity_req"] = s.requirements["capacity"]
                    self.user_data[u.id]["capacity_weight"] = s.requirement_weights["capacity"]
                    self.user_data[u.id]["capacity_drift"] = [0.0]*self.n_ttis
            
            # Data for users of slices with long term capacity requirements
            if s.requirements.get("long_term_capacity") is not None: 
                for u in s.users.values():
                    self.user_data[u.id]["long_term_capacity_req"] = s.requirements["long_term_capacity"]
                    self.user_data[u.id]["long_term_capacity_weight"] = s.requirement_weights["long_term_capacity"]
                    self.user_data[u.id]["long_term_capacity"] = [0.0]*self.n_ttis
                    self.user_data[u.id]["long_term_capacity_drift"] = [0.0]*self.n_ttis
            
            # Data for users of slices with latency requirements
            if s.requirements.get("latency") is not None: 
                for u in s.users.values():
                    self.user_data[u.id]["latency_req"] = s.requirements["latency"]
                    self.user_data[u.id]["latency_weight"] = s.requirement_weights["latency"]
                    self.user_data[u.id]["max_latency"] = u.buffer.max_lat
                    self.user_data[u.id]["pkt_size"] = u.pkt_generator.pkt_size
                    self.user_data[u.id]["buffer_size"] = int(u.buffer.buffer_size/u.pkt_generator.pkt_size)
                    self.user_data[u.id]["buffer"] = [0]*self.n_ttis
                    self.user_data[u.id]["drop_lat"] = [0]*self.n_ttis
                    self.user_data[u.id]["drop_buff"] = [0]*self.n_ttis
                    self.user_data[u.id]["sent"] = [0]*self.n_ttis
                    self.user_data[u.id]["part_sent"] = [0]*self.n_ttis
                    self.user_data[u.id]["pkts_per_lat"] = [{l: 0 for l in range(u.buffer.max_lat + 1)}]*self.n_ttis
                    self.user_data[u.id]["arrived"] = [self.arrived_pkts[u.id][t] for t in range(self.step, self.step+self.n_ttis)] # arrived_pkts[user][step]
                    self.user_data[u.id]["latency"] = [0.0]*self.n_ttis
                    self.user_data[u.id]["latency_drift"] = [0.0]*self.n_ttis
        
        # Calculating drifts assuming 0 capacity (no allocation) for all users 
        for u, data in self.user_data.items():
            self.update_data_user(ue_id=u, start=0, data=data)

    def update_data_intrasched(self, s: Slice, start:int, allocation:Dict[int, Dict[int, List[int]]], data:dict):
        """
        Updates the capacity data of users of the given slice considering the given allocation.
        Updates only for TTIs >= start. Simulates the intra-slice scheduler of the slice.
        """
        # Simulating Proportional Fair
        if isinstance(s.scheduler, intrasched.ProportionalFair):
            for t in range(start, self.n_ttis):
                # Update historical capacity and capacity (as we will recalculate)
                for u in s.users.values():
                    data[u.id]["capacity"][t] = 0
                    if t>0:
                        data[u.id]["historical_capacity"][t] = (1 - 1/self.metrics_window)*data[u.id]["historical_capacity"][t-1] + (1/self.metrics_window)*data[u.id]["capacity"][t-1]
                for r in allocation[t][s.id]:
                    # better_ue = max(s.users.keys(), key=lambda u: self.ue_tti_rbg_cap[u][t][r]/(data[u]["historical_capacity"][t]*(1.0 - 1.0/self.metrics_window) + data[u]["capacity"][t]/self.metrics_window))
                    better_ue = max(s.users.keys(), key=lambda u: self.ue_tti_rbg_cap[u][t][r]/data[u]["historical_capacity"][t])
                    data[better_ue]["capacity"][t] += self.ue_tti_rbg_cap[better_ue][t][r]
        
        # Simulating Maximum Throughput
        elif isinstance(s.scheduler, intrasched.MaximumThroughput):
            for t in range(start, self.n_ttis):
                # Update capacity (as we will recalculate)
                for u in s.users.values():
                    data[u.id]["capacity"][t] = 0
                for r in allocation[t][s.id]:
                    better_ue = max(s.users.keys(), key=lambda u: self.ue_tti_rbg_cap[u][t][r])
                    data[better_ue]["capacity"][t] += self.ue_tti_rbg_cap[better_ue][t][r]
        
        # Simulating Round-Robin
        elif isinstance(s.scheduler, intrasched.RoundRobin):
            users_ids = list(s.users.keys())
            offset = sum(len(allocation[t_][s.id]) for t_ in range(start)) % len(s.users)
            for t in range(start, self.n_ttis):
                # Update capacity (as we will recalculate)
                for u in s.users.values():
                    data[u.id]["capacity"][t] = 0
                for r in allocation[t][s.id]:
                    data[users_ids[offset]]["capacity"][t] += self.ue_tti_rbg_cap[users_ids[offset]][t][r]
                    offset = (offset + 1) % len(s.users)
    
    def update_data_user(self, ue_id:int, start:int, data:dict) -> None:
        """
        Updates the drift data of the given user considering the given data.
        Updates only for TTIs >= start. Returns the total accumulated drift at the last TTI.
        """

        for t in range(start, self.n_ttis):
            data["drift"][t] = 0
            data["total_drift"][t] = 0 if t == 0 else data["total_drift"][t-1]
            if data.get("capacity_req") is not None:
                data["capacity_drift"] = 0 if data["capacity"][t] >= data["capacity_req"] else (data["capacity_req"] - data["capacity"][t])/data["capacity_req"]
                data["drift"][t] += data["capacity_drift"]*data["capacity_weight"]
            if data.get("long_term_capacity_req") is not None:
                actual_window = t if t-self.metrics_window+1 < 0 else self.metrics_window
                data["long_term_capacity"][t] = sum(data["capacity"][t-actual_window:t+1])/(actual_window + 1)
                data["long_term_capacity_drift"] = 0 if data["long_term_capacity"][t] >= data["long_term_capacity_req"] else (data["long_term_capacity_req"] - data["long_term_capacity"][t])/data["long_term_capacity_req"]
                data["drift"][t] += data["long_term_capacity_drift"]*data["long_term_capacity_weight"]
            if data.get("latency_req") is not None:
                data["buffer"][t] = 0 if t == 0 else data["buffer"][t-1] + data["arrived"][t-1] - data["sent"][t-1] - data["drop_buff"][t-1] - data["drop_lat"][t-1]
                data["drop_buff"][t] = max(0, data["buffer"][t] + data["arrived"][t] - data["buffer_size"])
                data["sent"][t] = (
                    min(int(data["capacity"][t]*self.tti_lenght/data["pkt_size"]), data["buffer"][t] + data["arrived"][t] - data["drop_buff"][t])
                    if t == 0 else
                    min(
                        int(data["capacity"][t]*self.tti_lenght/data["pkt_size"] + data["part_sent"][t-1]),
                        data["buffer"][t] + data["arrived"][t] - data["drop_buff"][t]
                    )
                )
                for l in range(data["max_latency"]):
                    data["pkts_per_lat"][t][l] =  (
                        0 
                        if t-l < 0 else 
                        data["buffer"][t-l] + data["arrived"][t-l] - data["drop_buff"][t-l] - sum(data["sent"][t-l:t+1]) - sum(data["drop_lat"][t-l:t])
                    )
                data["drop_lat"][t] = data["pkts_per_lat"][t][data["max_latency"]]
                data["part_sent"][t] = (
                    0
                    if data["buffer"][t] + data["arrived"][t] - data["sent"][t] - data["drop_buff"][t] - data["drop_lat"][t] == 0 or data["drop_lat"][t] > 0 else
                    data["capacity"][t]*self.tti_lenght/data["pkt_size"] - data["sent"][t]
                )
                data["latency"][t] = max([l if data["pkts_per_lat"][t][l] > 0 else 0 for l in range(data["max_latency"])])
                data["latency_drift"] = 0 if data["latency"][t] <= data["latency_req"] else (data["latency"][t] - data["latency_req"])/(data["max_latency"]-data["latency_req"])
                data["drift"][t] += data["latency_drift"]*data["latency_weight"]
            data["total_drift"][t] += data["drift"][t]
        return data["total_drift"][-1]
    
    def update_data(self, new_allocation_decision:Tuple[int,int,int]):
        """
        Updates data of users given a new allocation decision.
        Updates only the data of users in the given slice and from the given TTI and after.
        """
        
        # Updating the allocation
        self.allocation[new_allocation_decision[2]][new_allocation_decision[1]].append(new_allocation_decision[0])

        # Simulating the intra-slice scheduler of the affected slice and updating the data of the affected users
        start = new_allocation_decision[2]
        s = new_allocation_decision[1]
        self.update_data_intrasched(s=self.slices[s], start=start, allocation=self.allocation, data=self.user_data)
        for u in self.slices[s].users.values():
            self.update_data_user(ue_id=u.id, start=start, data=self.user_data[u.id])
            

    def total_drift_for_allocation(self, r:int, s:int, t:int) -> float:
        """
        Calculates the total drift for all slices (accumulated drift at the last TTI).
        Leverages the accumulated drift already calculated for other slices and the previous TTIs.
        Only calculates the drift of users associated with the slice s and for the TTIs >= t.
        Does not update the data.
        """
        # Building the new allocation with the new RBG
        allocation = copy.deepcopy(self.allocation)
        allocation[t][s].append(r)

        # Simulating the intra-slice scheduler of the affected slice and updating the data of the affected users
        data = copy.deepcopy(self.user_data)
        self.update_data_intrasched(s=self.slices[s], start=t, allocation=allocation, data=data)
        for u in self.slices[s].users.values():
            self.update_data_user(ue_id=u.id, start=t, data=data[u.id])
        
        # Calculating the total drift for the allocation as the weighted average of the slices' total drifts 
        total_drift = sum(
            sum( # Getting the average drift from users in the same slice
                data[u.id]["total_drift"][-1]
                for u in s_.users.values()
            )/len(s_.users)*self.slice_weights[s_.id]
            for s_ in self.slices.values()
        )
        return total_drift

    def calculate_allocation(
        self,
        ue_tti_rbg_cap:Dict[int, Dict[int, Dict[int, float]]],
        arrived_pkts:Dict[int, Dict[int, int]],
        slices:Dict[int, Slice],
        rbgs:List[int]
    ) -> None:
        """
        Calculates the allocations for the slices at all TTIs.
        """
        
        # Initializing structures
        self.ue_tti_rbg_cap = ue_tti_rbg_cap
        self.arrived_pkts = arrived_pkts
        self.slices = slices
        self.rbgs = rbgs
        self.slice_weights = {s.id: s.weight for s in slices.values()}
        self.allocation = {t: {s.id: [] for s in slices.values()} for t in range(self.n_ttis)}
        self.create_user_data()

        # Solving the allocation for each window
        for start in range(0, self.n_ttis, self.window):
            end = min(start + self.window, self.n_ttis) # Exclusive

            # Selecting the best possible allocations at a time, until all RBGs are allocated or the total drift is zero
            possible_allocations = {(r,s,t) for r in self.rbgs for s in self.slices.keys() for t in range(start, end)}
            while len(possible_allocations) > 0:
                best_total_drift = None
                best_allocation = None
                count = 0
                for r,s,t in possible_allocations:
                    # print(f"Tested {count} possibilities out of {len(possible_allocations)}")
                    count += 1
                    total_drift = self.total_drift_for_allocation(r,s,t)
                    if total_drift == 0: # If reached zero total drift, it doesn't need more RBGs
                        self.allocation[t][s].append(r)
                        return
                    if best_total_drift is None or total_drift < best_total_drift:
                        best_total_drift = total_drift
                        best_allocation = (r,s,t)
                
                # Already selected the best allocation for this iteration
                for s in self.slices.values():
                    possible_allocations.remove((best_allocation[0], s.id, best_allocation[2]))
                print("Allocated", best_allocation)
                self.update_data(best_allocation)

    def schedule(self, slices: Dict[int, Slice], users: Dict[int, User], rbgs: List[int]) -> None:
        for s in slices.values():
            for rbg in self.allocation[self.step][s.id]:
                s.allocate_rbg(rbg)
    
    def advance_step(self) -> None:
        self.step += 1



class StepwiseDriftHeuristic(InterSliceScheduler):
    def __init__(
        self,
        metrics_window:int, # Time window (in TTIs) for long term capacity and proportional fair score
        step:int = 0,
    ) -> None:
        self.metrics_window = metrics_window
        self.step = step
        self.user_drift:Dict[int, float] = None # user_drift[user_id] = drift
        self.user_capacity:Dict[int, List[float]] = None # user_capacity[user_id][step] = capacity
        self.slice_scheduled_rbgs:Dict[int,int] = None # slice_scheduled_rbgs[slice_id] = n_rbgs allocated in this scheduling

    def calculate_user_drift_empty_allocation(self, u:User, requirements:Dict[str, float], requirement_weights:Dict[str, float]) -> float:
        drift = 0
        if "capacity" in requirements.keys() and requirements["capacity"] > 0:
            drift += requirement_weights["capacity"]
        if "long_term_capacity" in requirements.keys():
            actual_window = self.step+1 if self.step-self.metrics_window+1 < 0 else self.metrics_window
            ltc = sum(self.user_capacity[u.id][t] for t in range(self.step-actual_window+1, self.step))/actual_window
            drift += (requirements["long_term_capacity"]-ltc)/requirements["long_term_capacity"]*requirement_weights["long_term_capacity"] if ltc < requirements["long_term_capacity"] else 0
        if "latency" in requirements.keys():
            lat = u.buffer.oldest_pkt_lat
            drift += (lat-requirements["latency"])/(u.buffer.max_lat - requirements["latency"])*requirement_weights["latency"] if lat > requirements["latency"] else 0
        return drift
    
    def calculate_drift_reduction(self, s: Slice, r:int) -> Tuple[float, User, float, float]:
        
        # Simulating the intra-slice scheduler to know what user will receive the RBG and their capacity
        if isinstance(s.scheduler, intrasched.RoundRobin):
            user = list(s.users.values())[(s.scheduler.offset + self.slice_scheduled_rbgs[s.id])%len(s.users)]
        elif isinstance(s.scheduler, intrasched.MaximumThroughput):
            user = max(s.users.values(), key=lambda u: u.tti_rbg_cap[self.step][r])
        elif isinstance(s.scheduler, intrasched.ProportionalFair):
            if self.step == 0:
                user = max(
                    s.users.values(),
                    key=lambda u: 
                    u.tti_rbg_cap[self.step][r]/
                    (
                        s.scheduler.starting_historical_capacity
                        # s.scheduler.starting_historical_capacity*(1.0-1.0/s.scheduler.window)
                        # + (self.user_capacity[u.id][-1])/s.scheduler.window
                    )
                )
            else:
                user = max(
                    s.users.values(),
                    key=lambda u: 
                    u.tti_rbg_cap[self.step][r]/
                    (
                        s.scheduler.hist_cap[u.id]
                        # s.scheduler.hist_cap[u.id]*(1.0-1.0/s.scheduler.window)
                        # + (self.user_capacity[u.id][-1])/s.scheduler.window
                    )
                )
        capacity = self.user_capacity[user.id][-1] + user.tti_rbg_cap[self.step][r]
        
        user_drift = 0.0
        if "capacity" in s.requirements.keys() and capacity < s.requirements["capacity"]:
            user_drift += ((s.requirements["capacity"]-capacity)/s.requirements["capacity"])*s.requirement_weights["capacity"]
        if "long_term_capacity" in s.requirements.keys():
            actual_window = self.step+1 if self.step-self.metrics_window+1 < 0 else self.metrics_window
            ltc = (capacity + sum(self.user_capacity[user.id][t] for t in range(self.step-actual_window+1, self.step)))/actual_window
            if ltc < s.requirements["long_term_capacity"]:
                user_drift += ((s.requirements["long_term_capacity"]-ltc)/s.requirements["long_term_capacity"])*s.requirement_weights["long_term_capacity"]
        if "latency" in s.requirements.keys():
            delivered_pkts = capacity*user.buffer.TTI/user.buffer.pkt_size
            lat = 0
            for l in reversed(range(user.buffer.oldest_pkt_lat+1)):
                if user.buffer.buff[l] > delivered_pkts:
                    lat = l
                    break
                else:
                    delivered_pkts -= user.buffer.buff[l]
            if lat > s.requirements["latency"]:
                user_drift += ((lat-s.requirements["latency"])/(user.buffer.max_lat-s.requirements["latency"]))*s.requirement_weights["latency"]
        user_drift = user_drift/len(s.users)*s.weight
        reduction = self.user_drift[user.id] - user_drift
        return reduction, user, user_drift, capacity

    def schedule(self, slices: Dict[int, Slice], users: Dict[int, User], rbgs: List[int]) -> None:
        
        # Initializing structures
        if self.user_capacity is None:
            self.user_capacity = {u.id: [] for u in users.values()}
        for u in users.values():
            self.user_capacity[u.id].append(0.0)
        self.user_drift = {
            u.id: self.calculate_user_drift_empty_allocation(u, s.requirements, s.requirement_weights)/len(s.users)*s.weight
            for s in slices.values() for u in s.users.values()
        }
        available_rbgs = set(rbgs)
        self.slice_scheduled_rbgs = {s.id: 0 for s in slices.values()}

        #  Allocating RBGs  
        while len(available_rbgs) > 0: # Stop if all RBGs are allocated
            best_drift_reduction = None
            best_allocation = None
            best_affected_user = None
            best_affected_user_drift = None
            best_affected_user_capacity = None
            
            # If there is no drift, stop allocation to save resources
            total_drift = sum(self.user_drift.values())
            if total_drift == 0:
                return
            
            for r in available_rbgs:
                for s in slices.values(): # Choosing the RBG-Slice allocation that reduces the total drift the most
                    reduction, new_affected_user, new_affected_user_drift, new_affected_user_capacity = self.calculate_drift_reduction(s, r)
                    if best_drift_reduction is None or reduction > best_drift_reduction:
                        best_drift_reduction = reduction
                        best_allocation = (r, s)
                        best_affected_user = new_affected_user
                        best_affected_user_drift = new_affected_user_drift
                        best_affected_user_capacity = new_affected_user_capacity
            
            # If no reduction could be achieved in this iteration, but there is still drift, allocate some RBG to the slice with higher drift
            if best_drift_reduction == 0 and total_drift > 0:
                r = next(iter(available_rbgs)) # Pick any available RBG
                s = max(slices.values(), key=lambda s: sum(self.user_drift[u.id] for u in s.users.values()))
                best_allocation = (rbgs[r], s)
                best_drift_reduction, best_affected_user, best_affected_user_drift, best_affected_user_capacity = self.calculate_drift_reduction(s, rbgs[r])

            # Allocating the best RBG and updating data structures
            best_allocation[1].allocate_rbg(best_allocation[0])
            available_rbgs.remove(best_allocation[0])
            self.user_drift[best_affected_user.id] = best_affected_user_drift
            self.user_capacity[best_affected_user.id][-1] = best_affected_user_capacity
            self.slice_scheduled_rbgs[best_allocation[1].id] += 1

            # print("total_drift", sum(self.user_drift.values()))
            # for u, drift in self.user_drift.items():
            #     print(f"UE {u} drift = {drift}")
            # print(f"Allocated {sum(self.slice_scheduled_rbgs.values())} RBGs")
    
    def advance_step(self) -> None:
        self.step += 1

class OptimalScheduler(InterSliceScheduler):
    """
    Schedules following the allocation decisions saved from the optimization model solution.
    """
    
    def __init__(self, scenario:str, n_ttis:int, rbg_size:int, n_rbgs:int, n_ues:int, n_slices:int, time_limit, seed:int, step:int = 0) -> None:
        model_name = f"{n_ttis}ttis_{rbg_size}rbg_size_{n_rbgs}rbgs_{n_ues}ues_{n_slices}slices_{time_limit}_time_limit_{seed}_seed"
        with open(f"results/{scenario}/{model_name}.pickle", 'rb') as file:
           results = pickle.load(file)
        
        self.allocation:Dict[int, List[int]] = {} # allocation[slice_id] = rbg_ids
        for t in (results["T"]):
            self.allocation[t] = {}
            for s in results["S"]:
                self.allocation[t][s] = []
                for u in results["U"][s]:
                    for r in results["R"]:
                        if results["rho"][u,r,t] == 1:
                            self.allocation[t][s].append(r)
        self.step = step
    
    def schedule(self, slices: Dict[int, Slice], users: Dict[int, User], rbgs: List[int]) -> None:
        for s in slices.values():
            for r in self.allocation[self.step][s.id]:
                s.allocate_rbg(r)
    
    def advance_step(self) -> None:
        self.step += 1

class RadiosaberScheduler(InterSliceScheduler):
    def __init__(
        self,
        step:int = 0,
    ) -> None:
        self.step = step
        self.rbs_offset = None

    # Calculates the number of RBGs to allocate to each slice
    def slice_quota(self, slices: Dict[int, Slice], n_rbgs:int) -> Dict[int, int]:
        
        # Initializing data structures
        if self.rbs_offset is None:
            self.rbs_offset = {s.id: 0.0 for s in slices.values()}
        rbs_share = {}
        rbgs_quota = {}
        
        # Calculating how many RBGs each slice should receive
        for s in slices.values():
            rbs_share[s.id] = n_rbgs*s.weight + self.rbs_offset[s.id] # The offset includes the previous fractional part
            rbgs_quota[s.id] = int(rbs_share[s.id]) # Flooring the number of RBGs
        
        # Randomly distributing the remaining RBGs
        extra_rbgs = n_rbgs - sum(rbgs_quota.values())
        while extra_rbgs > 0:
            random_slice = random.choice(list(slices.keys()))
            rbgs_quota[random_slice] += 1
            extra_rbgs -= 1
        
        # Calculating the new offsets
        for s in slices.values():
            self.rbs_offset[s.id] = rbs_share[s.id] - rbgs_quota[s.id]
        
        return rbgs_quota

    # Simulates the intra-slice scheduler to get wich user should receive the RBG and their resulting capacity
    def get_intra_sched_user(self, s: Slice, r:int) -> Tuple[User, float]:
        if isinstance(s.scheduler, intrasched.RoundRobin):
            user = list(s.users.values())[(s.scheduler.offset + self.slice_scheduled_rbgs[s.id])%len(s.users)]
        elif isinstance(s.scheduler, intrasched.MaximumThroughput):
            user = max(s.users.values(), key=lambda u: u.tti_rbg_cap[self.step][r])
        elif isinstance(s.scheduler, intrasched.ProportionalFair):
            if self.step == 0:
                user = max(
                    s.users.values(),
                    key=lambda u: 
                    u.tti_rbg_cap[self.step][r]/s.scheduler.starting_historical_capacity
                )
            else:
                user = max(
                    s.users.values(),
                    key=lambda u: 
                    u.tti_rbg_cap[self.step][r]/s.scheduler.hist_cap[u.id]
                )
        return user, user.tti_rbg_cap[self.step][r]

    def schedule(self, slices: Dict[int, Slice], users: Dict[int, User], rbgs: List[int]) -> None:
        
        # Initializing structers
        available_rbgs = set(rbgs)
        self.slice_scheduled_rbgs = {s.id: 0 for s in slices.values()}

        # Getting the number of RBGs to allocate to each slice
        n_rbgs_per_slice = self.slice_quota(slices, len(rbgs))

        # Allocates every RBG
        while len(available_rbgs) > 0:
            best_allocation = None # (r,s)
            best_capacity = None
            best_user = None

            # Choosing the best RBG-Slice pair to allocate
            for s in slices.values():
                if n_rbgs_per_slice[s.id] == 0: # If the slice quota is already filled, ignore
                    continue
                for r in available_rbgs:
                    user, capacity = self.get_intra_sched_user(s, r)
                    if best_capacity is None or capacity > best_capacity:
                        best_allocation = (r,s.id)
                        best_capacity = capacity
                        best_user = user

            # Allocating the best RBG and updating data structures
            slices[best_allocation[1]].allocate_rbg(best_allocation[0])
            available_rbgs.remove(best_allocation[0])
            n_rbgs_per_slice[best_allocation[1]] -= 1
            self.slice_scheduled_rbgs[best_allocation[1]] += 1

    def advance_step(self) -> None:
        self.step += 1

class ModifiedRadiosaberScheduler(InterSliceScheduler):
    def __init__(
        self,
        metrics_folder:str,
        sim_name:str,
        seed:int,
        step:int = 0,
    ) -> None:
        self.step = step
        self.rbs_offset = None

        # Reading the simulation description (slices, users, etc.)
        with open(f"{metrics_folder}/{sim_name}_{seed}/description.json") as f:
            self.data_description = pd.read_json(f)

        # Reading the slice metrics
        self.slice_data = {}
        for slice_id in self.data_description["slices"].keys():
            self.slice_data[slice_id] = pd.read_csv(f"{metrics_folder}/{sim_name}_{seed}/slice_{slice_id}_metrics.csv")

    # Calculates the number of RBGs to allocate to each slice
    def slice_quota(self, slices: Dict[int, Slice], n_rbgs:int) -> Dict[int, int]:
        rbgs_quota = {}
        for slice in slices.values():
            rbgs_quota[slice.id] = self.slice_data[slice.id].loc[self.slice_data[slice.id]["step"] == self.step, "n_allocated_rbgs"].values[0]        
        return rbgs_quota

    # Simulates the intra-slice scheduler to get wich user should receive the RBG and their resulting capacity
    def get_intra_sched_user(self, s: Slice, r:int) -> Tuple[User, float]:
        if isinstance(s.scheduler, intrasched.RoundRobin):
            user = list(s.users.values())[(s.scheduler.offset + self.slice_scheduled_rbgs[s.id])%len(s.users)]
        elif isinstance(s.scheduler, intrasched.MaximumThroughput):
            user = max(s.users.values(), key=lambda u: u.tti_rbg_cap[self.step][r])
        elif isinstance(s.scheduler, intrasched.ProportionalFair):
            if self.step == 0:
                user = max(
                    s.users.values(),
                    key=lambda u: 
                    u.tti_rbg_cap[self.step][r]/s.scheduler.starting_historical_capacity
                )
            else:
                user = max(
                    s.users.values(),
                    key=lambda u: 
                    u.tti_rbg_cap[self.step][r]/s.scheduler.hist_cap[u.id]
                )
        return user, user.tti_rbg_cap[self.step][r]

    def schedule(self, slices: Dict[int, Slice], users: Dict[int, User], rbgs: List[int]) -> None:
        
        # Initializing structers
        available_rbgs = set(rbgs)
        self.slice_scheduled_rbgs = {s.id: 0 for s in slices.values()}

        # Getting the number of RBGs to allocate to each slice
        n_rbgs_per_slice = self.slice_quota(slices, len(rbgs))

        # Allocate the specified number of RBGs for each slice
        while len(available_rbgs) > 0 and sum(n_rbgs_per_slice.values()) > 0:
            best_allocation = None # (r,s)
            best_capacity = None
            best_user = None

            # Choosing the best RBG-Slice pair to allocate
            for s in slices.values():
                if n_rbgs_per_slice[s.id] == 0: # If the slice quota is already filled, ignore
                    continue
                for r in available_rbgs:
                    user, capacity = self.get_intra_sched_user(s, r)
                    if best_capacity is None or capacity > best_capacity:
                        best_allocation = (r,s.id)
                        best_capacity = capacity
                        best_user = user

            # Allocating the best RBG and updating data structures
            slices[best_allocation[1]].allocate_rbg(best_allocation[0])
            available_rbgs.remove(best_allocation[0])
            n_rbgs_per_slice[best_allocation[1]] -= 1
            self.slice_scheduled_rbgs[best_allocation[1]] += 1

    def advance_step(self) -> None:
        self.step += 1