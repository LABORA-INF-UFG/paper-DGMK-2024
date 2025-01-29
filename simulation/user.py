import numpy as np
from typing import List, Dict

from simulation.buffer import DiscreteBuffer
from simulation.pktgenerator import PktGenerator

class User:
    def __init__(
        self,
        id: int,
        step:int,
        TTI: float, # s
        tti_rbg_cap: Dict[int, Dict[int, float]], # tti_rbg_cap[tti][rbg] = capacity
        max_lat: int, # maximum latency in TTIs
        buffer_size: int, # bits
        pkt_size: int, # bits
        flow_type: str, # "poisson"
        flow_throughput: float, # bits/s
        rng: np.random.BitGenerator,
    ) -> None:
        self.id = id
        self.step = step
        self.tti_rbg_cap = tti_rbg_cap 
        self.buffer:DiscreteBuffer = DiscreteBuffer(
            TTI=TTI,
            max_lat=max_lat,
            buffer_size=buffer_size,
            pkt_size=pkt_size
        )
        self.pkt_generator:PktGenerator = PktGenerator(
            TTI=TTI,
            type=flow_type,
            pkt_size=pkt_size,
            throughput=flow_throughput,
            rng=rng
        )
        self.requirements:Dict[str, float] = None
        self.rbgs: List[int] = []
    
    def arrive_pkts(self):
        self.buffer.arrive_pkts(self.pkt_generator.generate_pkts())

    def transmit(self):
        self.buffer.transmit(capacity=sum([self.tti_rbg_cap[self.step][rbg] for rbg in self.rbgs]))

    def advance_step(self) -> None:
        self.buffer.advance_step()
        self.clear_rbg_allocation()
        self.step += 1

    def set_requirements(self, requirements: Dict[str, float]) -> None:
        self.requirements = requirements

    def set_requirements_weights(self, requirement_weights: Dict[str, float]) -> None:
        self.requirement_weights = requirement_weights

    def allocate_rbg(self, rbg:int) -> None:
        self.rbgs.append(rbg)
    
    def clear_rbg_allocation(self) -> None:
        self.rbgs: List[int] = []
