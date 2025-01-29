import numpy as np 
from typing import List

class DiscreteBuffer():
    def __init__(
        self,
        TTI: float, # s
        max_lat: int, # TTIs
        buffer_size: int, # bits
        pkt_size: int, # bits
    ) -> None:
        self.TTI = TTI
        self.max_lat = max_lat 
        self.buffer_size = buffer_size 
        self.pkt_size = pkt_size 
        self.buff:List[int] = [0]*(self.max_lat+1)
        self.partial_sent_pkt_bits:float = 0.0
        self.dropped_pkts_buffer_full:int = 0
        self.dropped_pkts_max_lat:int = 0
        self.arriv_pkts:int = 0
        self.sent_pkts:int = 0
        self.oldest_pkt_lat:int = 0

    def arrive_pkts(self, n_pkts: int) -> None:
        self.arriv_pkts = n_pkts
        dropped_bits:int = (self.arriv_pkts + sum(self.buff))*self.pkt_size - self.buffer_size
        self.dropped_pkts_buffer_full = 0 if dropped_bits <= 0 else int(np.ceil(dropped_bits/self.pkt_size))
        self.buff[0] += self.arriv_pkts - self.dropped_pkts_buffer_full
    
    def transmit(self, capacity:float) -> None:
        # Calculating the number of packets to send
        real_pkts = (capacity*self.TTI + self.partial_sent_pkt_bits)/self.pkt_size
        int_pkts = int(real_pkts)
        self.partial_sent_pkt_bits = (real_pkts - int_pkts)*self.pkt_size
        self.sent_pkts = 0
        
        # Sending packets
        for i in reversed(range(self.max_lat+1)):
            if self.buff[i] >= int_pkts:
                self.buff[i] -= int_pkts
                self.sent_pkts += int_pkts
                break
            else:
                int_pkts -= self.buff[i]
                self.sent_pkts += self.buff[i]
                self.buff[i] = 0
        
        # Calculating dropped packets
        self.dropped_pkts_max_lat = self.buff[-1]
        
        # Resetting partial_sent_pkt_bits if a partially sent packet is no longer in the buffer
        if self.dropped_pkts_max_lat > 0 or sum(self.buff) == 0:
            self.partial_sent_pkt_bits = 0
        
        # Calculating the latency of the oldest packet
        self.update_oldest_pkt_lat()
        
    def update_oldest_pkt_lat(self) -> None:
        last_lat = self.oldest_pkt_lat + 1
        if last_lat > self.max_lat:
            last_lat = self.max_lat
        self.oldest_pkt_lat = 0
        for l in reversed(range(last_lat + 1)):
            if self.buff[l] > 0:
                self.oldest_pkt_lat = l
                break
    
    def advance_step(self) -> None:
        for i in range(self.max_lat):
            self.buff[self.max_lat - i] = self.buff[self.max_lat - i - 1]
        self.buff[0] = 0
        self.update_oldest_pkt_lat()