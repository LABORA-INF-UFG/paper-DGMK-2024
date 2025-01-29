import numpy as np

class PktGenerator:
    def __init__(
        self,
        TTI: float, # s
        type: str, # "poisson"
        pkt_size: int, # bits
        throughput: float, # bits/s
        rng: np.random.BitGenerator
    ) -> None:
        self.type = type
        self.TTI = TTI
        self.pkt_size = pkt_size
        self.throughput = throughput
        self.rng = rng
        self.part_pkt_bits:float = 0.0

    def generate_pkts (self) -> int:
        if self.type == "poisson":
            bits = self.rng.poisson(self.throughput)*self.TTI + self.part_pkt_bits
        else:
            raise Exception("Flow type {} not supported".format(self.type))
        pkts = int(bits/self.pkt_size)
        self.part_pkt_bits = bits - pkts*self.pkt_size
        return pkts