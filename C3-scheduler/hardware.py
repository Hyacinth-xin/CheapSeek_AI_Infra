from typing import List


class HardwareCapabilities:
    def __init__(self):
        self.max_threads_per_block = 1024
        self.smem_bytes = 48 * 1024
        self.max_grid_dim = (2**31 - 1, 65535, 65535)
        self.num_sm = 128
        self.global_memory_bytes = 48 * 1024 * 1024 * 1024

    def supported_precisions(self) -> List[str]:
        return ["fp32", "fp16", "fp8", "fp4"]

    def max_block_size(self) -> int:
        return self.max_threads_per_block

    def shared_memory_size(self) -> int:
        return self.smem_bytes


hardware = HardwareCapabilities()
