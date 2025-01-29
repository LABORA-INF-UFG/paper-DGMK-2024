from typing import List, Dict
import numpy
import random
import math
import json
import numpy as np

from .SlicingModel import SlicingModel

class Experiment:
    def __init__(
        self,
        workers: int,
        slice_requirements: Dict[int, Dict[str, Dict[str, float]]],
        slice_demands: Dict[int, float],
        slice_pkt_size: Dict[int, int],
        slice_buffer_size: Dict[int, int],
        slice_weights: Dict[int, float],
        slice_max_latencies: Dict[int, int],
        coding_gain: float,
    ):
        self.workers=workers
        self.slice_requirements = slice_requirements
        self.slice_demands = slice_demands
        self.slice_pkt_size = slice_pkt_size
        self.slice_buffer_size = slice_buffer_size
        self.slice_weights = slice_weights
        self.slice_max_latencies = slice_max_latencies
        self.coding_gain = coding_gain
        
        # Reading the CQI to spectral efficiency mapping
        with open("config/cqi_to_spec_eff.json", "r") as f:
            data = json.load(f)
            self.cqi_to_spec_eff = {int(key): value for key, value in data.items()}

        # Reading the CQI to code rate mapping
        with open("config/cqi_to_code_rate.json", "r") as f:
            data = json.load(f)
            self.cqi_to_code_rate = {int(key): value for key, value in data.items()}

        # Reading the CQI to modulation order mapping
        with open("config/cqi_to_mod_ord.json", "r") as f:
            data = json.load(f)
            self.cqi_to_mod_ord = {int(key): value for key, value in data.items()}

        # Reading the CQI to SINR mapping
        with open("config/cqi_to_sinr.json", "r") as f:
            data = json.load(f)
            self.cqi_to_sinr = {int(key): value for key, value in data.items()}
        
    # Q function for BER calculation
    def q_func(self, x):
        return 0.5*math.erfc(x/math.sqrt(2))

    # Bit Error Rate
    def ber(self, mod_ord, spec_eff, code_rate, sinr): # sinr in dB
        ebn0 = 10**(sinr/10) / (mod_ord * code_rate) # From chatgpt TODO: FIND A SOURCE
        # ebn0 = spec_eff**-1 * (2.0**spec_eff - 1.0) # From Stallings, 5G Wireless, p. 542
        if mod_ord == 2: # QPSK
            return self.q_func(math.sqrt(2*ebn0)) # From Goldsmith, Wireless Communications, Table 6.1
        else: # M-QAM
            return 4/mod_ord * self.q_func(math.sqrt(3*ebn0*mod_ord/(2**mod_ord-1))) # From Goldsmith, Wireless Communications, Table 6.1

    # Packet Error Rate
    def per(self, mod_ord, spec_eff, pkt_size, code_rate, sinr):
        return 1.0 - (1.0 - self.ber(mod_ord, spec_eff, code_rate, sinr))**pkt_size # 1 or more bit error = packet error

    def generate_ue_tti_rbg_cap_and_per_big_blocks(
        self,
        ue_ids:List[int],
        rb_bandwidth:float,
        rbg_size:int,
        n_rbgs:int,
        n_ttis:int,
        ue_ids_per_slice:Dict[int, List[int]],
    ) -> Dict[int, Dict[int, Dict[int, float]]]:

        # Reading CQIs
        ue_cqi_path = "./cqi-traces-noise0/ue{}.log"
        ue_cqis = dict()
        for u in ue_ids:
            with open(ue_cqi_path.format(u), "r") as f:
                ue_cqis[u] = []
                for line in f:
                    row = list(map(int, line.strip().split()))
                    ue_cqis[u].append(row)
        
        # Transforming CQIs into spectral efficiencies
        ue_tti_rbg_cap = dict()
        for u in ue_ids:
            ue_tti_rbg_cap[u] = dict()
            for t in range(n_ttis):
                ue_tti_rbg_cap[u][t] = dict()
                for r in range(n_rbgs):
                    ue_tti_rbg_cap[u][t][r] = sum(rb_bandwidth * self.cqi_to_spec_eff[ue_cqis[u][t][r*rbg_size+i]] * self.cqi_to_code_rate[ue_cqis[u][t][r*rbg_size+i]] for i in range(rbg_size) ) # Average capacity for the RBG
        
        # Transforming CQIs into PERs
        ue_tti_rbg_per = dict()
        for u in ue_ids:
            ue_slice_id = next(key for key, value in ue_ids_per_slice.items() if u in value)
            ue_tti_rbg_per[u] = dict()
            for t in range(n_ttis):
                ue_tti_rbg_per[u][t] = dict()
                for r in range(n_rbgs):
                    ue_tti_rbg_per[u][t][r] = sum( # The RBG PER is the weighted average of the PERs of the RBs (weight = RB capacity)
                        self.per(
                            mod_ord=self.cqi_to_mod_ord[ue_cqis[u][t][r*rbg_size+i]],
                            spec_eff=self.cqi_to_spec_eff[ue_cqis[u][t][r*rbg_size+i]],
                            pkt_size=self.slice_pkt_size[ue_slice_id],
                            code_rate=self.cqi_to_code_rate[ue_cqis[u][t][r*rbg_size+i]],
                            sinr=self.cqi_to_sinr[ue_cqis[u][t][r*rbg_size+i]] + self.coding_gain
                        ) * self.cqi_to_spec_eff[ue_cqis[u][t][r*rbg_size+i]]*rb_bandwidth # PER * weight = PER * capacity = PER * spectral efficiency * bandwidth
                        for i in range(rbg_size)
                    )/sum(self.cqi_to_spec_eff[ue_cqis[u][t][r*rbg_size+i]]*rb_bandwidth for i in range(rbg_size))
                    # ue_tti_rbg_per[u][t][r] = min( # The RBG PER is the best PER of the RBs
                    #     self.per(
                    #         mod_ord=self.cqi_to_mod_ord[ue_cqis[u][t][r*rbg_size+i]],
                    #         spec_eff=self.cqi_to_spec_eff[ue_cqis[u][t][r*rbg_size+i]],
                    #         pkt_size=self.slice_pkt_size[ue_slice_id],
                    #         code_rate=self.cqi_to_code_rate[ue_cqis[u][t][r*rbg_size+i]],
                    #         sinr=self.cqi_to_sinr[ue_cqis[u][t][r*rbg_size+i]] + self.coding_gain
                    #     ) for i in range(rbg_size)
                    # )

        return ue_tti_rbg_cap, ue_tti_rbg_per
    
    def generate_ue_tti_rbg_cap_and_per_random_blocks(
        self,
        ue_ids:List[int],
        rb_bandwidth:float,
        n_rbgs:int,
        n_ttis:int,
        ue_ids_per_slice:Dict[int, List[int]],
    ) -> Dict[int, Dict[int, Dict[int, float]]]:
        
        # Reading CQIs
        ue_cqi_path = "./cqi-traces-noise0/ue{}.log"
        ue_cqis = dict()
        for u in ue_ids:
            with open(ue_cqi_path.format(u), "r") as f:
                ue_cqis[u] = []
                for line in f:
                    row = list(map(int, line.strip().split()))
                    ue_cqis[u].append(row)
        
        # Select RBGs randomly with a fixed size of 4 RBs
        rbgs = [i*4 for i in range(int(512/4))]
        random.shuffle(rbgs)
        rbgs = rbgs[:n_rbgs]

        # Transforming CQIs into spectral efficiencies
        ue_tti_rbg_cap = dict()
        for u in ue_ids:
            ue_tti_rbg_cap[u] = dict()
            for t in range(n_ttis):
                ue_tti_rbg_cap[u][t] = dict()
                for r in range(n_rbgs):
                    ue_tti_rbg_cap[u][t][r] = 4 * rb_bandwidth * self.cqi_to_spec_eff[ue_cqis[u][t][rbgs[r]]] * self.cqi_to_code_rate[ue_cqis[u][t][rbgs[r]]] # Capacity for 4 RBs
        
        # Transforming CQIs into PERs
        ue_tti_rbg_per = dict()
        for u in ue_ids:
            ue_slice_id = next(key for key, value in ue_ids_per_slice.items() if u in value)
            ue_tti_rbg_per[u] = dict()
            for t in range(n_ttis):
                ue_tti_rbg_per[u][t] = dict()
                for r in range(n_rbgs):
                    ue_tti_rbg_per[u][t][r] = self.per(
                        mod_ord=self.cqi_to_mod_ord[ue_cqis[u][t][rbgs[r]]],
                        spec_eff=self.cqi_to_spec_eff[ue_cqis[u][t][rbgs[r]]],
                        pkt_size=self.slice_pkt_size[ue_slice_id],
                        code_rate=self.cqi_to_code_rate[ue_cqis[u][t][rbgs[r]]],
                        sinr=self.cqi_to_sinr[ue_cqis[u][t][rbgs[r]]] + self.coding_gain
                    )

        return ue_tti_rbg_cap, ue_tti_rbg_per

    def generate_arrived_packets(
        self,
        rng: numpy.random.BitGenerator,
        slice_demands:Dict[int, float],
        n_ttis:int,
        tti_length:float,
        slice_pkt_size:Dict[int, int],
        slice_ids:List[int],
        ue_ids_per_slice:Dict[int, List[int]]
    ) -> Dict[int, List[float]]:
        
        arrived_pkts_per_user = {u: [] for s in slice_ids for u in ue_ids_per_slice[s]}
        partial_pkts_per_user = {u: 0.0 for s in slice_ids for u in ue_ids_per_slice[s]}
        for t in range(n_ttis): # The RNG calls must happen in the same order of the simulation
            for s in slice_ids:
                for u in ue_ids_per_slice[s]:
                    num_pkts = rng.poisson(slice_demands[s])*tti_length/slice_pkt_size[s] + partial_pkts_per_user[u]
                    partial_pkts_per_user[u] = num_pkts - int(num_pkts)
                    arrived_pkts_per_user[u].append(int(num_pkts))
        return arrived_pkts_per_user

    def execute_experiment(
        self,
        seed:int,
        n_rbgs:int,
        rbg_size:int,
        n_ttis: int,
        has_per:bool,
        aggregation_method:str,
        slice_n_ues: Dict[int,int],
        slice_intra_schedulings: Dict[int,str],
        resource_minimization:bool,
        time_limit:int,
        tti_length:float,
        rb_bandwidth:float,
        window:int,
        rbg_grouping_method:str,
    ):
        
        # Resetting the randomness seed
        random.seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
        
        # Selecting UEs randomly
        n_ues = sum(slice_n_ues.values()) # Max number of UEs = 158
        ue_ids = list(range(158))
        np.random.shuffle(ue_ids)
        ue_ids = ue_ids[:n_ues]

        # Setting users to slices
        slice_ids = list(slice_n_ues.keys())
        ue_ids_per_slice = dict()
        included_ues = 0
        for s in slice_ids:
            ue_ids_per_slice[s] = ue_ids[included_ues:included_ues+slice_n_ues[s]]
            included_ues += slice_n_ues[s]

        # Reading CQI traces
        if rbg_grouping_method == "random_block":
            ue_tti_rbg_cap, ue_tti_rbg_per = self.generate_ue_tti_rbg_cap_and_per_random_blocks(
                ue_ids=ue_ids,
                rb_bandwidth=rb_bandwidth,
                n_rbgs=n_rbgs,
                n_ttis=n_ttis,
                ue_ids_per_slice=ue_ids_per_slice,
            )
        elif rbg_grouping_method == "big_block":
            ue_tti_rbg_cap, ue_tti_rbg_per = self.generate_ue_tti_rbg_cap_and_per_big_blocks(
                ue_ids=ue_ids,
                rb_bandwidth=rb_bandwidth,
                rbg_size=rbg_size,
                n_rbgs=n_rbgs,
                n_ttis=n_ttis,
                ue_ids_per_slice=ue_ids_per_slice,
            )
        
        # Setting the number of arrived packets for each user at each step
        arrived_pkts_per_user = self.generate_arrived_packets(
            rng=rng,
            slice_demands=self.slice_demands,
            n_ttis=n_ttis,
            tti_length=tti_length,
            slice_pkt_size=self.slice_pkt_size,
            slice_ids=slice_ids,
            ue_ids_per_slice=ue_ids_per_slice
        )
        
        # Setting the error time window at each step for each UE to calculate the PER
        error_window_start = dict()
        for s in slice_ids:
            if self.slice_requirements[s].get("per") is None:
                continue
            for u in ue_ids_per_slice[s]:
                pkt_window = int(self.slice_requirements[s]["per"]["req"]**-1) # If PER req is 10^-3, the window is 1000 packets
                error_window_start[u] = dict()
                for t in range(n_ttis):
                    t_ = t 
                    pkt_sum = 0
                    while t_ >=0 and pkt_sum < pkt_window: # We start from the current TTI and go back until we reach the packet window
                        pkt_sum += arrived_pkts_per_user[u][t_]
                        t_ -= 1
                    error_window_start[u][t] = t_ + 1
        
        # Setting the initial historical capacity for each UE as their requirements
        ue_initial_hist_cap = dict()
        for s in slice_ids:
            for u in ue_ids_per_slice[s]:
                if self.slice_requirements[s].get("cap") is not None:
                    ue_initial_hist_cap[u] = self.slice_requirements[s]["cap"]["req"]
                elif self.slice_requirements[s].get("ltc") is not None:
                    ue_initial_hist_cap[u] = self.slice_requirements[s]["ltc"]["req"]
                else:
                    ue_initial_hist_cap[u] = 1e6 # Default value = 1 Mbps

        # Initializing the model parameters
        model_name = f"{n_ttis}ttis_{rbg_size}rbg_size_{n_rbgs}rbgs_{n_ues}ues_{len(slice_ids)}slices_{time_limit}_time_limit_{seed}_seed"
        model = SlicingModel(
            model_name=model_name,
            steps=n_ttis,
            window=window, # For averaging the long-term capacity
            epsilon=1e-6, # Small error margin
            big_M=1e9, # Big value
            TTI_length=tti_length, # TTI length = 1 ms
            rbg_bandwidth=rbg_size*rb_bandwidth,
            slice_ids=slice_ids,
            ue_ids_per_slice=ue_ids_per_slice,
            slice_max_latencies=self.slice_max_latencies,
            arrived_pkts_per_user=arrived_pkts_per_user,
            n_rbgs=n_rbgs,
            ue_tti_rbg_cap=ue_tti_rbg_cap,
            slices_requirements=self.slice_requirements, # Slice id : metric name :{requirement:value, weight:value}
            slice_weights=self.slice_weights,
            pkt_size_per_slice=self.slice_pkt_size,
            buffer_size_per_slice=self.slice_buffer_size,
            slice_intra_schedulings=slice_intra_schedulings,
            error_window_start=error_window_start,
            user_drift_aggregation_method=aggregation_method,
            ue_initial_hist_cap=ue_initial_hist_cap,
            ue_tti_rbg_per=ue_tti_rbg_per,
            resource_minimization=resource_minimization,
            has_per=has_per,
        )

        # Building the model
        print("Building model...")
        building_time = model.build(search_type="IterativeDiving") # DepthFirst, Restart, MultiPoint, IterativeDiving, Neighborhood, or Auto
        print(f"Building time = {building_time} seconds")

        # Running the model
        print("Solving model")
        solution = model.run(
            cpooptimizer_bin_path="/opt/ibm/ILOG/CPLEX_Studio221/cpoptimizer/bin/x86-64_linux/cpoptimizer",
            time_limit=time_limit,
            workers=self.workers
        )

        # Getting results
        status = solution.get_solve_status()
        results = None
        if status in ["Optimal", "Feasible"]:
            results = model.get_results()
            results["solve_status"] = solution.get_solve_status()
        return model_name, solution, results
