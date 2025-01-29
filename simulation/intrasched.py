from abc import ABC, abstractmethod
from typing import Dict, List

from simulation.user import User

class IntraSliceScheduler(ABC):
    @abstractmethod
    def schedule(self, rbgs:List[int], users=Dict[int, User]) -> None:
        raise Exception("Called abstract IntraSliceScheduler method")
    
    @abstractmethod
    def advance_step(self) -> None:
        raise Exception("Called abstract IntraSliceScheduler method")

class RoundRobin(IntraSliceScheduler):
    def __init__(self,offset: int = 0) -> None:
        self.offset = offset
    
    def schedule(self, rbgs:List[int], users=Dict[int, User]) -> None:
        ids: List[User] = list(users.keys())
        for r in rbgs:
            users[ids[self.offset]].allocate_rbg(r)
            self.offset = (self.offset + 1) % len(ids)
    
    def advance_step(self) -> None:
        pass

class MaximumThroughput(IntraSliceScheduler):
    def __init__(
            self,
            ue_tti_rbg_cap:Dict[int, Dict[int, Dict[int, float]]],
            step: int = 0,
        ) -> None:
        self.ue_tti_rbg_cap = ue_tti_rbg_cap
        self.step = step
    
    def score(self, u:int, r:int) -> float:
        return self.ue_tti_rbg_cap[u][self.step][r]

    def schedule(self, rbgs:List[int], users=Dict[int, User]) -> None:
        for r in rbgs:
            better_ue = max(users.keys(), key=lambda u: self.score(u,r))
            users[better_ue].allocate_rbg(r)

    def advance_step(self) -> None:
        self.step += 1

class ProportionalFair(IntraSliceScheduler):
    def __init__(
            self,
            ue_tti_rbg_cap:Dict[int, Dict[int, Dict[int, float]]],
            step: int = 0,
            window: int = 10,
            starting_historical_capacity: float = 1.0, # bits/s
        ) -> None:
        self.ue_tti_rbg_cap = ue_tti_rbg_cap
        self.step = step
        self.window = window
        self.starting_historical_capacity = starting_historical_capacity
        self.hist_cap: Dict[int, float] = {}
        self.actual_cap: Dict[int, float] = {}
    
    def score(self, u:int, r:int) -> float:
        if u not in self.hist_cap:
            self.hist_cap[u] = self.starting_historical_capacity
        if u not in self.actual_cap:
            self.actual_cap[u] = 0.0
        # return self.ue_tti_rbg_cap[u][self.step][r]/(self.hist_cap[u]*(1.0-1.0/self.window) + self.actual_cap[u]/self.window)
        return self.ue_tti_rbg_cap[u][self.step][r]/self.hist_cap[u]

    def schedule(self, rbgs:List[int], users=Dict[int, User]) -> None:
        for r in rbgs:
            better_ue = max(users.keys(), key=lambda u: self.score(u,r))
            users[better_ue].allocate_rbg(r)
            self.actual_cap[better_ue] += self.ue_tti_rbg_cap[better_ue][self.step][r]

    def advance_step(self) -> None:
        self.step += 1
        for u in self.hist_cap.keys():
            self.hist_cap[u] = self.hist_cap[u] * (1.0 - 1.0/self.window) + self.actual_cap[u] / self.window
        for u in self.actual_cap.keys():
            self.actual_cap[u] = 0.