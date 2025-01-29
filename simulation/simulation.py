import numpy as np
from typing import Dict, List
import time

from simulation.intersched import InterSliceScheduler
from simulation.slice import Slice
from simulation.user import User
from simulation.intrasched import IntraSliceScheduler

class Simulation:
    def __init__(
        self,
        experiment_name: str,
        scheduler: InterSliceScheduler,
        numerology: int,
        rbgs: List[int],
        rng: np.random.BitGenerator,
        ue_tti_rbg_cap: Dict[int, Dict[int, Dict[int, float]]], # ue_tti_rbg_cap[user][tti][rbg] = capacity
    ) -> None:
        if numerology < 0 or numerology > 4:
            raise Exception("Numerology = {} is not valid for the simulation (must be 0, 1, 2, 3 or 4)".format(numerology))
        self.experiment_name = experiment_name
        self.scheduler = scheduler
        self.rbgs = rbgs
        self.rng = rng
        self.ue_tti_rbg_cap = ue_tti_rbg_cap
        self.TTI:float = 2**-numerology * 1e-3 # s
        self.step = 0
        self.slices: Dict[int, Slice] = {}
        self.users: Dict[int, User] = {}
        self.generated_arrived_pkts:Dict[int, Dict[int, int]] = None # generated_arrived_pkts[user][tti] = n_pkts

    def add_slice(
        self,
        slice_id: int,
        slice_type: str,
        requirements: Dict[str, float],
        requirement_weights: Dict[str, float],
        weight: float,
        intra_scheduler: IntraSliceScheduler
    ) -> None:
        if slice_id in self.slices:
            raise Exception("Slice {} already exists".format(slice_id))
        self.slices[slice_id] = Slice(
            id=slice_id,
            type=slice_type,
            requirements=requirements,
            requirement_weights=requirement_weights,
            weight=weight,
            scheduler=intra_scheduler,
        )

    def add_users(
        self,
        slice_id: int,
        user_ids: List[int],
        max_lat: int, # maximum latency in TTIs
        buffer_size: int, # bits
        pkt_size: int, # bits
        flow_type: str, # "poisson"
        flow_throughput: float, # bits/s
    ) -> None:
        for u in user_ids:
            new_user = User(
                id=u,
                step=self.step,
                TTI=self.TTI,
                tti_rbg_cap=self.ue_tti_rbg_cap[u],
                max_lat=max_lat,
                buffer_size=buffer_size,
                pkt_size=pkt_size,
                flow_type=flow_type,
                flow_throughput=flow_throughput,
                rng=self.rng,
            )
            self.users[u] = new_user
            self.slices[slice_id].assign_user(user=new_user)
    
    def generate_arrived_pkts(self, n_ttis:int) -> Dict[int, Dict[int, int]]:
        self.generated_arrived_pkts = dict()
        for u in self.users.values():
            self.generated_arrived_pkts[u.id] = dict()
        for t in range(n_ttis):
            for u in self.users.values():
                self.generated_arrived_pkts[u.id][t] = u.pkt_generator.generate_pkts()
        return self.generated_arrived_pkts
        
    def arrive_packets(self) -> None:
        if self.generated_arrived_pkts is None:
            for u in self.users.values():
                u.arrive_pkts()
        else:
            for u in self.users.values():
                u.buffer.arrive_pkts(n_pkts=self.generated_arrived_pkts[u.id][self.step])
    
    def schedule_rbgs(self) -> None:
        start = time.time()
        self.scheduler.schedule(
            slices=self.slices,
            users=self.users,
            rbgs=self.rbgs
        )
        self.scheduler_time = time.time() - start
        for s in self.slices.values():
            s.schedule_rbgs() 

    def transmit(self) -> None:
        for u in self.users.values():
            u.transmit()
    
    def advance_step(self) -> None:
        for u in self.users.values():
            u.advance_step()
        for s in self.slices.values():
            s.advance_step()
        self.scheduler.advance_step()
        self.step += 1