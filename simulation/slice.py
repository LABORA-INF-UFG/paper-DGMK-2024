from typing import Dict, List

from simulation.user import User
from simulation.intrasched import IntraSliceScheduler

class Slice:
    def __init__(
        self,
        id: int,
        type: str,
        requirements: Dict[str, float],
        requirement_weights: Dict[str, float],
        weight: float,
        scheduler: IntraSliceScheduler,
    ) -> None:
        self.id = id
        self.type = type
        self.requirements = requirements
        self.requirement_weights = requirement_weights
        self.weight = weight
        self.scheduler = scheduler
        self.users: Dict[int, User] = dict()
        self.rbgs: List[int] = []

    def assign_user(self, user: User) -> None:
        if user.id in self.users.values():
            raise Exception("User {} is already assigned to slice {}".format(user.id, self.id))
        user.set_requirements(requirements=self.requirements)
        user.set_requirements_weights(requirement_weights=self.requirement_weights)
        self.users[user.id] = user

    def allocate_rbg(self, rbg:int) -> None:
        self.rbgs.append(rbg)
    
    def clear_rbg_allocation(self) -> None:
        self.rbgs: List[int] = []

    def schedule_rbgs(self) -> None:
        self.scheduler.schedule(rbgs=self.rbgs, users=self.users)
    
    def advance_step(self) -> None:
        self.clear_rbg_allocation()
        self.scheduler.advance_step()