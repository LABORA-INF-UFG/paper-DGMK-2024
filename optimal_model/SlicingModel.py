import time
from docplex.cp.model import CpoModel
from docplex.cp.parameters import CpoParameters
from docplex.cp import modeler
from os.path import exists
from types import SimpleNamespace
from typing import List, Dict
from math import sqrt

class SlicingModel:
    def __init__(
            self,
            model_name:str,
            steps: int,
            TTI_length: float,
            slice_ids: List[int],
            ue_ids_per_slice: Dict[int, List[int]],
            slice_max_latencies: Dict[int, int],
            arrived_pkts_per_user: Dict[int, Dict[int, int]],
            n_rbgs: int,
            ue_tti_rbg_cap: Dict[int, Dict[int, Dict[int, float]]],
            ue_tti_rbg_per: Dict[int, Dict[int, Dict[int, float]]],
            rbg_bandwidth: float,
            window: int,
            slices_requirements: Dict[int, Dict[str, Dict[str, float]]],
            epsilon: float,
            big_M: float,
            slice_weights: Dict[int, float],
            pkt_size_per_slice: Dict[int, int], # in bits
            buffer_size_per_slice: Dict[int, int], # in packets
            slice_intra_schedulings: Dict[int, str], # intra-slice scheduling algorithms for each slice
            error_window_start: Dict[int,Dict[int,int]], # EW[u][t]
            user_drift_aggregation_method: str,
            ue_initial_hist_cap: Dict[int, float], # initial historical capacity for each user
            resource_minimization: bool,
            has_per: bool
        ):
        self.model_name = model_name
        self.steps = steps
        self.TTI_length = TTI_length
        self.slice_ids = slice_ids
        self.ue_ids_per_slice = ue_ids_per_slice
        self.slice_max_latencies = slice_max_latencies
        self.arrived_pkts_per_user = arrived_pkts_per_user
        self.n_rbgs = n_rbgs
        self.ue_tti_rbg_cap = ue_tti_rbg_cap
        self.ue_tti_rbg_per = ue_tti_rbg_per
        self.rbg_bandwidth = rbg_bandwidth
        self.window = window
        self.slices_requirements = slices_requirements
        self.epsilon = epsilon
        self.big_M = big_M
        self.slice_weights = slice_weights
        self.buffer_size_per_slice = buffer_size_per_slice
        self.pkt_size_per_slice = pkt_size_per_slice
        self.slice_intra_schedulings = slice_intra_schedulings
        self.error_window_start = error_window_start
        self.user_drift_aggregation_method = user_drift_aggregation_method
        self.model = None
        self.ue_initial_hist_cap = ue_initial_hist_cap
        self.resource_minimization = resource_minimization
        self.has_per = has_per

    def run(self, cpooptimizer_bin_path:str, time_limit:int=None, solution_limit:int=None, workers:int=None):
        """
        Runs the built constraint optimization model.
        """
        self.solution = self.model.solve(
            agent="local",
            LogVerbosity="Verbose",
            log_output=True,
            execfile=cpooptimizer_bin_path,
            TimeLimit=time_limit,
            SolutionLimit=solution_limit,
            LogPeriod=int(5e5),
            Workers=workers
        )
        return self.solution

    def build(self, search_type:str="Auto"):
        """
        Builds the constraint optimization model.
        """
        
        building_model_start_time = time.time()

        m = CpoModel(name=self.model_name)
        m.set_parameters(CpoParameters(SearchType=search_type))

        # ------------------------- 
        # SETS
        # -------------------------

        # T set - set of steps
        T = range(self.steps)

        # S set - set of slices
        S = self.slice_ids

        # U[s] sets - set of users for each slice s
        U = self.ue_ids_per_slice

        # L[s] sets - set of latencies for each slice s
        L = {s: range(self.slice_max_latencies[s] + 1) for s in S}

        # R set - set of resource blocks
        R = range(self.n_rbgs)

        # ------------------------- 
        # VARIABLES
        # -------------------------
        
        # rho[u,i,t] - (DECISION VARIABLE) resource block i allocated to user u at step t
        m.rho = m.binary_var_dict( name = "rho",
            keys=[(u, r, t) for s in S for u in U[s] for r in R for t in T],
        )

        # m.k_floor[u,t] - (FLOORING VARIABLE) how many packets from user u could deliver with the current capacity at step t
        m.k_floor = m.integer_var_dict( name = "k_floor",
            keys=[(u, t) for s in S for u in U[s] for t in T],
        )

        # ------------------------- 
        # INDICATOR VARIABLES
        # -------------------------

        # m.ind_f_cap_is_zero[u,t] - indicate if the required SLA capacity is achieved for user u at step t
        m.ind_f_cap_is_zero = m.binary_var_dict( name = "ind_f_cap_is_zero",
            keys=[(u,t) for s in S for u in U[s] for t in T if self.slices_requirements[s].get("cap") is not None],
        )

        # m.ind_f_ltc_is_zero[u,t] - indicate if the required SLA long-term throughput is achieved for user u at step t
        m.ind_f_ltc_is_zero = m.binary_var_dict( name = "ind_f_ltc_is_zero",
            keys=[(u,t) for s in S for u in U[s] for t in T if self.slices_requirements[s].get("ltc") is not None],
        )

        # m.ind_f_lat_is_zero[u,t] - indicate if the required SLA latency is achieved for user u at step t
        m.ind_f_lat_is_zero = m.binary_var_dict( name = "ind_f_lat_is_zero",
            keys=[(u,t) for s in S for u in U[s] for t in T if self.slices_requirements[s].get("lat") is not None],
        )
        
        if self.has_per:
            # m.ind_f_per_is_zero[u,t] - indicate if the required SLA packet error rate is achieved for user u at step t
            m.ind_f_per_is_zero = m.binary_var_dict( name = "ind_f_per_is_zero",
                keys=[(u,t) for s in S for u in U[s] for t in T if self.slices_requirements[s].get("per") is not None],
            )

        # m.a[t] - indicate scarce scenario for each step t
        if self.resource_minimization:
            m.a = m.binary_var_dict( name = "a",
                keys=[t for t in T],
            )
                
        # m.ind_remains_pkt[u,t] - there will still be at least 1 packet in user u's buffer at the end of step t
        m.ind_remains_pkt = m.binary_var_dict( name = "ind_remains_pkt",
            keys=[(u, t) for s in S for u in U[s] for t in T],
        )

        # m.ind_has_space[u,t] - indicate if there is space in the buffer for the arrived packets
        m.ind_has_space = m.binary_var_dict( name = "ind_has_space",
            keys=[(u, t) for s in S for u in U[s] for t in T],
        )

        # m.ind_remains_pkt_l[u,l,t] - there will still be at least 1 packet in user u's buffer that waited for l steps at the end of step t
        m.ind_remains_pkt_l = m.binary_var_dict( name = "ind_remains_pkt_l",
            keys=[(u, t, l) for s in S for u in U[s] for l in L[s] for t in T if t-l >= 0],
        )

        # ------------------------- 
        # EXPRESSIONS
        # -------------------------
        
        # c[u,t] - capacity (bits/s) for user u at step t
        c = {
            (u,t): sum(m.rho[u,r,t] * self.ue_tti_rbg_cap[u][t][r] for r in R) for s in S for u in U[s] for t in T
        }

        # b[u,t] - how many packets in user u's buffer at the beginning of step t
        # df[u,t] - how many packets were dropped for buffer full at step t
        # k_[u,t] - fraction of a partially delivered packet at step t
        # k[u,t] - how many packets user u delivered at step t
        # dl[u,t] - how many packets were dropped for surpassing the maximum latency at the end of step t
        # p[u,t,l] - how many packets arrived at step t-l and are still in the buffer at the end of step t
        b = dict()
        df = dict()
        k = dict()
        k_ = dict()
        dl = dict()
        p = dict()
        for s in S:
            for u in U[s]:
                # Setting k_[u,-1] to express the k_floor at the first step (t=0)
                k_[u,-1] = 0

                for t in T:
                    b[u,t] = 0 if t == 0 else b[u,t-1] + self.arrived_pkts_per_user[u][t-1] - k[u,t-1] - df[u,t-1] - dl[u,t-1] # Buffer starts empty
                    df[u,t] = (1-m.ind_has_space[u,t])*(b[u,t] + self.arrived_pkts_per_user[u][t] - self.buffer_size_per_slice[s])
                    k[u,t] = modeler.min(m.k_floor[u,t], b[u,t] + self.arrived_pkts_per_user[u][t] - df[u,t])
                    k_[u,t] = (
                        m.ind_remains_pkt[u,t] * (c[u,t]*self.TTI_length/self.pkt_size_per_slice[s] - k[u,t])
                        if t < L[s][-1]
                        else m.ind_remains_pkt[u,t] * (1-m.ind_remains_pkt_l[u,t, L[s][-1]]) * (c[u,t]*self.TTI_length/self.pkt_size_per_slice[s] - k[u,t])
                    )
                    
                    # Defining p[u,t,l]
                    for l in L[s]:
                        if t-l < 0:
                            p[u,t,l] = 0
                        else:
                            p[u,t,l] = m.ind_remains_pkt_l[u,t,l] * (
                                b[u,t-l]
                                + self.arrived_pkts_per_user[u][t-l]
                                - df[u,t-l]
                                - sum(dl[u,t_] for t_ in T[t-l:t])
                                - sum(k[u,t_] for t_ in T[t-l:t+1])
                            )

                    dl[u,t] = p[u,t, L[s][-1]]

        # h[u,t] - historical exponential weighted moving average capacity (bits/s) for user u at step t
        h = dict()
        for s in S:
            for u in U[s]:
                for t in T:
                    if t == 0:
                        h[u,0] = self.ue_initial_hist_cap[u]
                    else:
                        h[u,t] = (
                            (1-1/self.window) * h[u,t-1] 
                            + 1/self.window * c[u,t-1]
                        )

        # pf_score[u,r,t] - proportional fair score for user u and RBG r at step t
        # pf_score = {
        #     (u,r,t):(self.ue_tti_rbg_cap[u][t][r])/((1-1/self.window)*h[u,t] + (1/self.window) * sum(m.rho[u,r_,t] * self.ue_tti_rbg_cap[u][t][r_] for r_ in R[:r]))
        #     for s in S for u in U[s] for r in R for t in T
        # }
        pf_score = {
            (u,r,t):(self.ue_tti_rbg_cap[u][t][r])/h[u,t] # Trying a simpler proportional fair that calculates the score once per step
            for s in S for u in U[s] for r in R for t in T
        }

        # mt_score[u,r,t] - maximum throughput score for user u and RBG r at step t
        mt_score = {
            (u,r,t): self.ue_tti_rbg_cap[u][t][r]
            for s in S for u in U[s] for r in R for t in T
        }

        # cap[u,t] - capacity (bits/s) for user u at step t
        cap = {
            (u, t): c[u, t]
            for s in S for u in U[s] for t in T
            if self.slices_requirements[s].get("cap") is not None
        }

        # ltc[u,t] - long term capacity (bits/s) for user u at step t
        ltc = {
            (u, t):sum(c[u, t_] for t_ in T[:t+1]) / (t + 1)
            if t < self.window
            else sum(c[u, t_] for t_ in T[t - self.window + 1:t + 1]) / self.window
            for s in S for u in U[s] for t in T
            if self.slices_requirements[s].get("ltc") is not None
        }

        # lat[u,t] - average buffer latency (ms) for user u at step t
        lat = {
            (u, t): modeler.max([l*m.ind_remains_pkt_l[u,t,l] for l in L[s] if t >= l])
            for s in S for u in U[s] for t in T
            if self.slices_requirements[s].get("lat") is not None
        }

        if self.has_per:
            # per[u,t] - packet error rate at step t
            per = {
                (u, t): sum(m.rho[u,r,t]*self.ue_tti_rbg_cap[u][t][r]*self.ue_tti_rbg_per[u][t][r]for r in R)/(self.epsilon + sum(m.rho[u,r,t]*self.ue_tti_rbg_cap[u][t][r] for r in R))
                for s in S for u in U[s] for t in T
                if self.slices_requirements[s].get("per") is not None
            }

        # f_cap[u,t] - SLA drift in capacity for user u at step t
        f_cap = {
            (u,t):(1-m.ind_f_cap_is_zero[u,t])*(self.slices_requirements[s]["cap"]["req"]-cap[u, t])/self.slices_requirements[s]["cap"]["req"]
            for s in S for u in U[s] for t in T
            if self.slices_requirements[s].get("cap") is not None
        }

        # f_ltc[u,t] - SLA drift in long-term capacity for user u at step t
        f_ltc = {
            (u,t):(1-m.ind_f_ltc_is_zero[u,t])*(self.slices_requirements[s]["ltc"]["req"]-ltc[u, t])/self.slices_requirements[s]["ltc"]["req"]
            for s in S for u in U[s] for t in T
            if self.slices_requirements[s].get("ltc") is not None
        }

        # f_lat[u,t] - SLA drift in latency for user u at step t
        f_lat = {
            (u,t):(1-m.ind_f_lat_is_zero[u,t])*(lat[u, t]-self.slices_requirements[s]["lat"]["req"])/(self.slice_max_latencies[s]-self.slices_requirements[s]["lat"]["req"])
            for s in S for u in U[s] for t in T
            if self.slices_requirements[s].get("lat") is not None
        }

        if self.has_per:
            # f_per[t] - SLA drift in packet error rate for user u at step t
            f_per = {
                (u,t):(1-m.ind_f_per_is_zero[u,t])*(per[u, t]-self.slices_requirements[s]["per"]["req"])/(1-self.slices_requirements[s]["per"]["req"])
                for s in S for u in U[s] for t in T
                if self.slices_requirements[s].get("per") is not None
            }

        if self.has_per:
            # f_usr[u,t] - SLA drift for user u at step t
            f_usr = {
                (u, t): sum([
                    f_cap[u, t] * self.slices_requirements[s]["cap"]["weight"] if "cap" in self.slices_requirements[s] else 0,
                    f_ltc[u, t] * self.slices_requirements[s]["ltc"]["weight"] if "ltc" in self.slices_requirements[s] else 0,
                    f_lat[u, t] * self.slices_requirements[s]["lat"]["weight"] if "lat" in self.slices_requirements[s] else 0,
                    f_per[u, t] * self.slices_requirements[s]["per"]["weight"] if "per" in self.slices_requirements[s] else 0
                ]) for s in S for u in U[s] for t in T
            }
        else:
            # f_usr[u,t] - SLA drift for user u at step t
            f_usr = {
                (u, t): sum([
                    f_cap[u, t] * self.slices_requirements[s]["cap"]["weight"] if "cap" in self.slices_requirements[s] else 0,
                    f_ltc[u, t] * self.slices_requirements[s]["ltc"]["weight"] if "ltc" in self.slices_requirements[s] else 0,
                    f_lat[u, t] * self.slices_requirements[s]["lat"]["weight"] if "lat" in self.slices_requirements[s] else 0
                ]) for s in S for u in U[s] for t in T
            }

        # f_slice[t] - SLA drift for slice s at step t
        if self.user_drift_aggregation_method == "average":
            f_slice = {
                (s, t): sum(f_usr[u, t] for u in U[s])/len(U[s])
                for s in S for t in T 
            }
        elif self.user_drift_aggregation_method == "average+variance":
            f_slice = {
                (s, t): sum(f_usr[u, t] for u in U[s])/len(U[s]) + sum((f_usr[u, t] - sum(f_usr[u, t] for u in U[s])/len(U[s]))**2 for u in U[s])/(len(U[s])-1)
                for s in S for t in T
            }
        elif self.user_drift_aggregation_method == "average+std_dev":
            f_slice = {
                (s, t): sum(f_usr[u, t] for u in U[s])/len(U[s]) + sqrt(sum((f_usr[u, t] - sum(f_usr[u, t] for u in U[s])/len(U[s]))**2 for u in U[s])/(len(U[s])-1))
                for s in S for t in T
            }
        else:
            raise ValueError("Invalid user drift aggregation method. Choose between 'average', 'average+variance' and 'average+std_dev'.")

        # f[t] - total SLA drift at step t
        f = {
            t: sum(f_slice[s, t]*self.slice_weights[s] for s in S)
            for t in T
        }

        # ------------------------- 
        # OBJECTIVE
        # -------------------------

        # Primary goal: if it is not possible to achieve drift = 0, minimize the drift
        # Secondary goal: if it is possible to achieve drift = 0, minimize the resource allocation
        if self.resource_minimization:
            m.minimize(
                sum(
                    m.a[t] * (f[t] + 1) + (1-m.a[t]) * sum (m.rho[u, r, t] for r in R for s in S for u in U[s] )/len(R)
                    for t in T
                )/len(T)
            )
        else: # Only minimize the drift
            m.minimize(
                sum(
                    f[t]
                    for t in T
                )/len(T)
            )

        # ------------------------- 
        # CONSTRAINTS
        # -------------------------

        # Each RBG r must be allocated to at most one user u at each step t
        for r in R:
            for t in T:
                m.add(
                    sum(m.rho[u, r, t] for s in S for u in U[s]) <= 1
                )
        
        # m.k_floor[u,t] must be the integer number of packets that can be delivered by user u at step t
        for s in S:
            for u in U[s]:
                for t in T:
                    m.add(
                        m.k_floor[u,t] <= c[u,t]*self.TTI_length/self.pkt_size_per_slice[s] + k_[u,t-1]
                    )
                    m.add(
                        m.k_floor[u,t] + 1 >= c[u,t]*self.TTI_length/self.pkt_size_per_slice[s] + k_[u,t-1] + self.epsilon
                    )

        # INTRA-SLICE SCHEDULING CONSTRAINTS
        for s in S:
            if self.slice_intra_schedulings[s] == "proportional_fair":
                for r in R:
                    for t in T:
                        # Enforcing the selection of the user with the highest score
                        max_score = sum(pf_score[u_,r,t] * m.rho[u_,r,t] for u_ in U[s])
                        for u in U[s]:
                            rbg_allocated_to_slice = sum(m.rho[u_,r,t] for u_ in U[s])
                            m.add(
                                max_score >= pf_score[u,r,t] * rbg_allocated_to_slice
                            )
            elif self.slice_intra_schedulings[s] == "maximum_throughput":
                for r in R:
                    for t in T:
                        # Enforcing the selection of the user with the highest score
                        max_score = sum(mt_score[u_,r,t] * m.rho[u_,r,t] for u_ in U[s])
                        for u in U[s]:
                            rbg_allocated_to_slice = sum(m.rho[u_,r,t] for u_ in U[s])
                            m.add(
                                max_score >= mt_score[u,r,t] * rbg_allocated_to_slice
                            )
            elif self.slice_intra_schedulings[s] == "round_robin":
                # The difference in number of allocated resource blocks between two users u1 and u2 from the same slice s in step t must be at most 1
                for u1 in U[s]:
                    for u2 in U[s]:
                        for t in T:
                            if u1 != u2:
                                m.add(
                                    sum(m.rho[u1, r, t] for r in R) - sum(m.rho[u2, r, t] for r in R) <= 1
                                )
                # Forcing the round robin prioritization (treats the list of users as a circle)
                for t in T:
                    for j in range(len(U[s])-1):
                        u1 = U[s][(t+j)%len(U[s])] # Shifts the beggining of the circle every step
                        u2 = U[s][(t+j+1)%len(U[s])]
                        m.add(
                            sum(m.rho[u1, r, t] for r in R) >= sum(m.rho[u2, r, t] for r in R)
                        )
        
        # ------------------------- 
        # EXTRA CONSTRAINTS
        # -------------------------

        # EXTRA CONSTRAINT - Maximum number of RBGs available
        for t in T:
            m.add(
                sum(m.rho[u,r,t] for r in R for s in S for u in U[s]) <= len(R)
            )

        # EXTRA CONSTRAINT - k_floor is non-negative
        for s in S:
            for u in U[s]:
                for t in T:
                    m.add(
                        m.k_floor[u,t] >= 0
                    )

        if self.resource_minimization:
            # EXTRA CONSTRAINT - Restricting the objective as <= 2
            m.add(
                sum(
                    m.a[t] * (f[t] + 1) + (1-m.a[t]) * sum (m.rho[u, r, t] for r in R for s in S for u in U[s] )/len(R)
                    for t in T
                )/len(T) <= 2
            )

            # EXTRA CONSTRAINT - Restricting the objective as >= 0
            m.add(
                sum(
                    m.a[t] * (f[t] + 1) + (1-m.a[t]) * sum (m.rho[u, r, t] for r in R for s in S for u in U[s] )/len(R)
                    for t in T
                )/len(T) >= 0
            )
            
            # REMOVED BECAUSE IT DEGRADES THE PER AND PROPORTIONAL FAIR
            # # EXTRA CONSTRAINT - Enforcing the use of all RBGs in scarce scenarios
            # for t in T:
            #     m.add(
            #         sum(m.rho[u,r,t] for r in R for s in S for u in U[s]) >= m.a[t]*len(R)
            #     )
        else:
            # EXTRA CONSTRAINT - Restricting the objective as <= 1
            m.add(
                sum(
                    f[t]
                    for t in T
                )/len(T) <= 1
            )

            # EXTRA CONSTRAINT - Restricting the objective as >= 0
            m.add(
                sum(
                    f[t]
                    for t in T
                )/len(T) >= 0
            )
            
            
            # EXTRA CONSTRAINT - Enforcing the use of all RBGs if has drift
            for t in T:
                m.add(
                    sum(m.rho[u,r,t] for r in R for s in S for u in U[s]) >= len(R)*m.a[t]
                )
        

        # ------------------------- 
        # INDICATOR CONSTRAINTS
        # -------------------------
        
        # Contraints for ind_var = 1 <-> expression >= constant
        def add_ind_constr_ge(model, expression, constant, ind_var, upper_bound, lower_bound, epsilon):
            model.add(expression + lower_bound*ind_var >= lower_bound + constant)
            model.add(expression - (upper_bound + epsilon)*ind_var <= constant - epsilon)

        # Contraints for ind_var = 1 <-> expression <= constant
        def add_ind_constr_le(model, expression, constant, ind_var, upper_bound, lower_bound, epsilon):
            model.add(expression + upper_bound*ind_var <= upper_bound + constant)
            model.add(expression - (lower_bound - epsilon)*ind_var >= constant + epsilon)
        
        # m.ind_has_space[u,t] must indicate if there is space in the buffer for the arrived packets
        upper_bound = 1 + sum(self.arrived_pkts_per_user[u][t] for s in S for u in U[s] for t in T) + sum(b[u,0] for s in S for u in U[s])
        lower_bound = - self.buffer_size_per_slice[s] -1
        for s in S:
            for u in U[s]:
                for t in T:
                    add_ind_constr_le(
                        model=m,
                        expression=b[u,t] + self.arrived_pkts_per_user[u][t],
                        constant=self.buffer_size_per_slice[s],
                        ind_var=m.ind_has_space[u,t],
                        upper_bound=upper_bound,
                        lower_bound=lower_bound,
                        epsilon=self.epsilon
                    )

        # m.ind_remains_pkt_l[u,t,l] must indicate if there is a remaining packet at u's buffer that waited for l steps at step t
        upper_bound = 1 + sum(self.arrived_pkts_per_user[u][t] for s in S for u in U[s] for t in T) + sum(b[u,0] for s in S for u in U[s])
        for s in S:
            for u in U[s]:
                for l in L[s]:
                    for t in T:
                        if t-l < 0:
                            continue
                        add_ind_constr_ge(
                            model=m,
                            expression=b[u,t-l] + self.arrived_pkts_per_user[u][t-l] - (df[u,t-l] + sum(dl[u,t_] for t_ in T[t-l:t]) + sum(k[u,t_] for t_ in T[t-l:t+1])),
                            constant=1,
                            ind_var=m.ind_remains_pkt_l[u,t,l],
                            upper_bound=upper_bound,
                            lower_bound=-upper_bound-1,
                            epsilon=self.epsilon
                        )

        # m.ind_remains_pkt[u,t] - must indicate if at least 1 packet will be in user u's buffer at the end of step t
        upper_bound = 1 + sum(self.arrived_pkts_per_user[u][t] for s in S for u in U[s] for t in T) + sum(b[u,0] for s in S for u in U[s])
        for s in S:
            for u in U[s]:
                for t in T:
                    add_ind_constr_ge(
                        model=m,
                        expression=b[u,t] + self.arrived_pkts_per_user[u][t] - k[u,t] - df[u,t] - dl[u,t],
                        constant=1,
                        ind_var=m.ind_remains_pkt[u,t],
                        upper_bound=upper_bound,
                        lower_bound=0-1,
                        epsilon=self.epsilon
                    )

        if self.resource_minimization:
            # m.a[t] must indicate f[t] > 0, i.e. at least 1 zero drift indicator is 1
            for t in T:
                ind_cap_has_drift_list = [1 - m.ind_f_cap_is_zero[u,t] for s in S for u in U[s] if self.slices_requirements[s].get("cap") is not None]
                ind_ltc_has_drift_list = [1 - m.ind_f_ltc_is_zero[u,t] for s in S for u in U[s] if self.slices_requirements[s].get("ltc") is not None]
                ind_lat_has_drift_list = [1 - m.ind_f_lat_is_zero[u,t] for s in S for u in U[s] if self.slices_requirements[s].get("lat") is not None]
                if self.has_per:
                    ind_per_has_drift_list = [1 - m.ind_f_per_is_zero[u,t] for s in S for u in U[s] if self.slices_requirements[s].get("per") is not None]
                    expression = sum(ind_cap_has_drift_list) + sum(ind_ltc_has_drift_list) + sum(ind_lat_has_drift_list) + sum(ind_per_has_drift_list),
                    upper_bound = len(ind_cap_has_drift_list) + len(ind_ltc_has_drift_list) + len(ind_lat_has_drift_list) + len(ind_per_has_drift_list)
                else:
                    expression = sum(ind_cap_has_drift_list) + sum(ind_ltc_has_drift_list) + sum(ind_lat_has_drift_list)
                    upper_bound = len(ind_cap_has_drift_list) + len(ind_ltc_has_drift_list) + len(ind_lat_has_drift_list)
                add_ind_constr_ge(
                    model=m,
                    expression=expression,
                    constant=1,
                    ind_var=m.a[t],
                    upper_bound=upper_bound,
                    lower_bound=0-1,
                    epsilon=self.epsilon
                )

        # SLA requirement indicators
        for s in S:
            # m.ind_f_cap_is_zero[u,t] must indicate if the required SLA capacity is achieved for user u at step t
            if self.slices_requirements[s].get("cap") is not None:
                upper_bound = sum(m.rho[u,r,t] * self.ue_tti_rbg_cap[u][t][r] for s in S for u in U[s] for t in T for r in R)
                lower_bound = - self.slices_requirements[s]["cap"]["req"]
                for u in U[s]:
                    for t in T:
                        add_ind_constr_ge(
                            model=m,
                            expression=cap[u,t],
                            constant=self.slices_requirements[s]["cap"]["req"],
                            ind_var=m.ind_f_cap_is_zero[u,t],
                            upper_bound=upper_bound,
                            lower_bound=lower_bound,
                            epsilon=self.epsilon
                        )
            
            # m.ind_f_ltc_is_zero[u,t] must indicate if the required SLA long-term throughput is achieved for user u at step t
            if self.slices_requirements[s].get("ltc") is not None:
                upper_bound = sum(m.rho[u,r,t] * self.ue_tti_rbg_cap[u][t][r] for s in S for u in U[s] for t in T for r in R)
                lower_bound = - self.slices_requirements[s]["ltc"]["req"]
                for u in U[s]:
                    for t in T:
                        add_ind_constr_ge(
                            model=m,
                            expression=ltc[u,t],
                            constant=self.slices_requirements[s]["ltc"]["req"],
                            ind_var=m.ind_f_ltc_is_zero[u,t],
                            upper_bound=upper_bound,
                            lower_bound=lower_bound,
                            epsilon=self.epsilon
                        )
            
            # m.ind_f_lat_is_zero[u,t] must indicate if the required SLA latency is achieved for user u at step t
            if self.slices_requirements[s].get("lat") is not None:
                upper_bound = sum(len(L[s]) for s in S)
                lower_bound = - self.slices_requirements[s]["lat"]["req"]
                for u in U[s]:
                    for t in T:
                        add_ind_constr_le(
                            model=m,
                            expression=lat[u,t],
                            constant=self.slices_requirements[s]["lat"]["req"],
                            ind_var=m.ind_f_lat_is_zero[u,t],
                            upper_bound=upper_bound,
                            lower_bound=lower_bound,
                            epsilon=self.epsilon
                        )

            # m.ind_f_per_is_zero[u,t] must indicate if the required SLA packet error rate is achieved for user u at step t
            if self.has_per and self.slices_requirements[s].get("per") is not None:
                upper_bound = 1 + 1 # Upper bound for per[u,t] - self.slices_requirements[s]["per"]["req"]
                lower_bound = 0 - 1 # Lower bound for per[u,t] - self.slices_requirements[s]["per"]["req"]
                for u in U[s]:
                    for t in T:
                        add_ind_constr_le(
                            model=m,
                            expression=per[u,t],
                            constant=self.slices_requirements[s]["per"]["req"],
                            ind_var=m.ind_f_per_is_zero[u,t],
                            upper_bound=upper_bound,
                            lower_bound=lower_bound,
                            epsilon=self.epsilon
                        )

        # Saving the built model
        self.model = m

        # Returning the building time
        building_model_end_time = time.time()
        return building_model_end_time - building_model_start_time
    
    def get_results(self):
        """
        Returns the results of the optimization model.
        """

        # Sets
        T = [t for t in range(self.steps)]
        S = self.slice_ids
        U = self.ue_ids_per_slice
        L = {s: [l for l in range(self.slice_max_latencies[s] + 1)] for s in S}
        R = [r for r in range(self.n_rbgs)]

        # Variables
        m = SimpleNamespace()
        m.rho = {(u,r,t): self.solution[self.model.rho[u,r,t]] for s in S for u in U[s] for r in R for t in T}
        if self.resource_minimization:
            m.a = {t: self.solution[self.model.a[t]] for t in T}
        m.k_floor = {(u,t): self.solution[self.model.k_floor[u,t]] for s in S for u in U[s] for t in T}
        m.ind_has_space = {(u,t): self.solution[self.model.ind_has_space[u,t]] for s in S for u in U[s] for t in T}
        m.ind_remains_pkt = {(u,t): self.solution[self.model.ind_remains_pkt[u,t]] for s in S for u in U[s] for t in T}
        m.ind_remains_pkt_l = {(u,t,l): self.solution[self.model.ind_remains_pkt_l[u,t,l]] for s in S for u in U[s] for l in L[s] for t in T if t >= l}
        m.ind_f_cap_is_zero = {(u,t): self.solution[self.model.ind_f_cap_is_zero[u,t]] for s in S for u in U[s] for t in T if self.slices_requirements[s].get("cap") is not None}
        m.ind_f_ltc_is_zero = {(u,t): self.solution[self.model.ind_f_ltc_is_zero[u,t]] for s in S for u in U[s] for t in T if self.slices_requirements[s].get("ltc") is not None}
        m.ind_f_lat_is_zero = {(u,t): self.solution[self.model.ind_f_lat_is_zero[u,t]] for s in S for u in U[s] for t in T if self.slices_requirements[s].get("lat") is not None}
        if self.has_per:
            m.ind_f_per_is_zero = {(u,t): self.solution[self.model.ind_f_per_is_zero[u,t]] for s in S for u in U[s] for t in T if self.slices_requirements[s].get("per") is not None}
        
        
        # Expressions
       
        # c[u,t] - capacity (bits/s) for user u at step t
        c = {
            (u,t): sum(m.rho[u,r,t] * self.ue_tti_rbg_cap[u][t][r] for r in R) for s in S for u in U[s] for t in T
        }

        # b[u,t] - how many packets in user u's buffer at the beginning of step t
        # df[u,t] - how many packets were dropped for buffer full at step t
        # k_[u,t] - fraction of a partially delivered packet at step t
        # k[u,t] - how many packets user u delivered at step t
        # dl[u,t] - how many packets were dropped for surpassing the maximum latency at the end of step t
        # p[u,t,l] - how many packets arrived at step t-l and are still in the buffer at the end of step t
        b = dict()
        df = dict()
        k = dict()
        k_ = dict()
        dl = dict()
        p = dict()
        for s in S:
            for u in U[s]:
                # Setting k_[u,-1] to express the k_floor at the first step (t=0)
                k_[u,-1] = 0

                for t in T:
                    b[u,t] = 0 if t == 0 else b[u,t-1] + self.arrived_pkts_per_user[u][t-1] - k[u,t-1] - df[u,t-1] - dl[u,t-1] # Buffer starts empty
                    df[u,t] = (1-m.ind_has_space[u,t])*(b[u,t] + self.arrived_pkts_per_user[u][t] - self.buffer_size_per_slice[s])
                    k[u,t] = modeler.min(m.k_floor[u,t], b[u,t] + self.arrived_pkts_per_user[u][t] - df[u,t])
                    k_[u,t] = (
                        m.ind_remains_pkt[u,t] * (c[u,t]*self.TTI_length/self.pkt_size_per_slice[s] - k[u,t])
                        if t < L[s][-1]
                        else m.ind_remains_pkt[u,t] * (1-m.ind_remains_pkt_l[u,t, L[s][-1]]) * (c[u,t]*self.TTI_length/self.pkt_size_per_slice[s] - k[u,t])
                    )
                    
                    # Defining p[u,t,l]
                    for l in L[s]:
                        if t-l < 0:
                            p[u,t,l] = 0
                        else:
                            p[u,t,l] = m.ind_remains_pkt_l[u,t,l] * (
                                b[u,t-l]
                                + self.arrived_pkts_per_user[u][t-l]
                                - df[u,t-l]
                                - sum(dl[u,t_] for t_ in T[t-l:t])
                                - sum(k[u,t_] for t_ in T[t-l:t+1])
                            )

                    dl[u,t] = p[u,t, L[s][-1]]

        # h[u,t] - historical exponential weighted moving average capacity (bits/s) for user u at step t
        h = dict()
        for s in S:
            for u in U[s]:
                for t in T:
                    if t == 0:
                        h[u,0] = self.ue_initial_hist_cap[u]
                    else:
                        h[u,t] = (
                            (1-1/self.window) * h[u,t-1] 
                            + 1/self.window * c[u,t-1]
                        )

        # pf_score[u,r,t] - proportional fair score for user u and RBG r at step t
        pf_score = {
            (u,r,t):(self.ue_tti_rbg_cap[u][t][r])/((1-1/self.window)*h[u,t] + (1/self.window) * sum(m.rho[u,r_,t] * self.ue_tti_rbg_cap[u][t][r_] for r_ in R[:r]))
            for s in S for u in U[s] for r in R for t in T
        }

        # mt_score[u,r,t] - maximum throughput score for user u and RBG r at step t
        mt_score = {
            (u,r,t): self.ue_tti_rbg_cap[u][t][r]
            for s in S for u in U[s] for r in R for t in T
        }

        # cap[u,t] - capacity (bits/s) for user u at step t
        cap = {
            (u, t): c[u, t]
            for s in S for u in U[s] for t in T
            if self.slices_requirements[s].get("cap") is not None
        }

        # ltc[u,t] - long term capacity (bits/s) for user u at step t
        ltc = {
            (u, t):sum(c[u, t_] for t_ in T[:t+1]) / (t + 1)
            if t < self.window
            else sum(c[u, t_] for t_ in T[t - self.window + 1:t + 1]) / self.window
            for s in S for u in U[s] for t in T
            if self.slices_requirements[s].get("ltc") is not None
        }

        # lat[u,t] - average buffer latency (ms) for user u at step t
        lat = {
            (u, t): modeler.max([l*m.ind_remains_pkt_l[u,t,l] for l in L[s] if t >= l])
            for s in S for u in U[s] for t in T
            if self.slices_requirements[s].get("lat") is not None
        }

        # per[u,t] - packet error rate at step t
        per = {
            (u, t): sum(m.rho[u,r,t]*self.ue_tti_rbg_cap[u][t][r]*self.ue_tti_rbg_per[u][t][r]for r in R)/(self.epsilon + sum(m.rho[u,r,t]*self.ue_tti_rbg_cap[u][t][r] for r in R))
            for s in S for u in U[s] for t in T
            if self.slices_requirements[s].get("per") is not None
        }

        # f_cap[u,t] - SLA drift in capacity for user u at step t
        f_cap = {
            (u,t):(cap[u, t] < self.slices_requirements[s]["cap"]["req"])*(self.slices_requirements[s]["cap"]["req"]-cap[u, t])/self.slices_requirements[s]["cap"]["req"]
            for s in S for u in U[s] for t in T
            if self.slices_requirements[s].get("cap") is not None
        }

        # f_ltc[u,t] - SLA drift in long-term capacity for user u at step t
        f_ltc = {
            (u,t):(ltc[u, t] < self.slices_requirements[s]["ltc"]["req"])*(self.slices_requirements[s]["ltc"]["req"]-ltc[u, t])/self.slices_requirements[s]["ltc"]["req"]
            for s in S for u in U[s] for t in T
            if self.slices_requirements[s].get("ltc") is not None
        }

        # f_lat[u,t] - SLA drift in latency for user u at step t
        f_lat = {
            (u,t):(lat[u, t] > self.slices_requirements[s]["lat"]["req"])*(lat[u, t]-self.slices_requirements[s]["lat"]["req"])/(self.slice_max_latencies[s]-self.slices_requirements[s]["lat"]["req"])
            for s in S for u in U[s] for t in T
            if self.slices_requirements[s].get("lat") is not None
        }

        # f_per[t] - SLA drift in packet error rate for user u at step t
        f_per = {
            (u,t):(per[u, t] > self.slices_requirements[s]["per"]["req"])*(per[u, t]-self.slices_requirements[s]["per"]["req"])/(1-self.slices_requirements[s]["per"]["req"])
            for s in S for u in U[s] for t in T
            if self.slices_requirements[s].get("per") is not None
        }

        if self.has_per:
            # f_usr[u,t] - SLA drift for user u at step t
            f_usr = {
                (u, t): sum([
                    f_cap[u, t] * self.slices_requirements[s]["cap"]["weight"] if "cap" in self.slices_requirements[s] else 0,
                    f_ltc[u, t] * self.slices_requirements[s]["ltc"]["weight"] if "ltc" in self.slices_requirements[s] else 0,
                    f_lat[u, t] * self.slices_requirements[s]["lat"]["weight"] if "lat" in self.slices_requirements[s] else 0,
                    f_per[u, t] * self.slices_requirements[s]["per"]["weight"] if "per" in self.slices_requirements[s] else 0
                ]) for s in S for u in U[s] for t in T
            }
        else:
            # f_usr[u,t] - SLA drift for user u at step t
            f_usr = {
                (u, t): sum([
                    f_cap[u, t] * self.slices_requirements[s]["cap"]["weight"] if "cap" in self.slices_requirements[s] else 0,
                    f_ltc[u, t] * self.slices_requirements[s]["ltc"]["weight"] if "ltc" in self.slices_requirements[s] else 0,
                    f_lat[u, t] * self.slices_requirements[s]["lat"]["weight"] if "lat" in self.slices_requirements[s] else 0,
                ]) for s in S for u in U[s] for t in T
            }
        
        # f_slice[t] - SLA drift for slice s at step t
        if self.user_drift_aggregation_method == "average":
            f_slice = {
                (s, t): sum(f_usr[u, t] for u in U[s])/len(U[s])
                for s in S for t in T 
            }
        elif self.user_drift_aggregation_method == "average+variance":
            f_slice = {
                (s, t): sum(f_usr[u, t] for u in U[s])/len(U[s]) + sum((f_usr[u, t] - sum(f_usr[u, t] for u in U[s])/len(U[s]))**2 for u in U[s])/(len(U[s])-1)
                for s in S for t in T
            }
        elif self.user_drift_aggregation_method == "average+std_dev":
            f_slice = {
                (s, t): sum(f_usr[u, t] for u in U[s])/len(U[s]) + sqrt(sum((f_usr[u, t] - sum(f_usr[u, t] for u in U[s])/len(U[s]))**2 for u in U[s])/(len(U[s])-1))
                for s in S for t in T
            }
        else:
            raise ValueError("Invalid user drift aggregation method. Choose between 'average', 'average+variance' and 'average+std_dev'.")

        # f[t] - total SLA drift at step t
        f = {
            t: sum(f_slice[s, t]*self.slice_weights[s] for s in S)
            for t in T
        }

        # a[t] - indicate scarce scenario for each step t
        a = {
            t: (f[t] > 0)
            for t in T
        }

        # for r in R:
        #     for t in T:      
        #         max_score = sum(pf_score[u_,r,t] * m.rho[u_,r,t] for u_ in U[s])
        #         print ("Max score:", max_score)
        #         for u in U[s]:
        #             rbg_allocated_to_slice = sum(m.rho[u_,r,t] for u_ in U[s])
        #             print("Other side:", pf_score[u,r,t] * rbg_allocated_to_slice, "Respects:", max_score >= pf_score[u,r,t] * rbg_allocated_to_slice)

        if self.resource_minimization:
            return {
                "rho": m.rho,
                "b": b,
                "T": T,
                "S": S,
                "U": U,
                "L": {s: L[s][-1] for s in S},
                "R": R,
                "p": p,
                "df": df,
                "dl": dl,
                "pf_score": pf_score,
                "mt_score": mt_score,
                "c": c,
                "k_floor": m.k_floor,
                "k_": k_,
                "k": k,
                "cap": cap,
                "ltc": ltc,
                "lat": lat,
                "per": per,
                "f_cap": f_cap,
                "f_ltc": f_ltc,
                "f_lat": f_lat,
                "f_per": f_per,
                "f_usr": f_usr,
                "f_slice": f_slice,
                "f": f,
                "a": a,
                "m.a": m.a,
                "objective": self.solution.get_objective_value(),
                "gap": self.solution.get_objective_gap(),
            }
        else:
            return {
                "rho": m.rho,
                "b": b,
                "T": T,
                "S": S,
                "U": U,
                "L": {s: L[s][-1] for s in S},
                "R": R,
                "p": p,
                "df": df,
                "dl": dl,
                "pf_score": pf_score,
                "mt_score": mt_score,
                "c": c,
                "k_floor": m.k_floor,
                "k_": k_,
                "k": k,
                "cap": cap,
                "ltc": ltc,
                "lat": lat,
                "per": per,
                "f_cap": f_cap,
                "f_ltc": f_ltc,
                "f_lat": f_lat,
                "f_per": f_per,
                "f_usr": f_usr,
                "f_slice": f_slice,
                "f": f,
                "a": a,
                "objective": self.solution.get_objective_value(),
                "gap": self.solution.get_objective_gap(),
            }