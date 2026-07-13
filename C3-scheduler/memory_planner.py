from typing import Dict, List, Tuple, Optional
from graph import Graph, NodeInfo
from hardware import HardwareCapabilities as HardwareSpec
from kernel import KernelInstance


class MemoryRegion:
    def __init__(self, name: str, size_bytes: int, dtype: str, is_input: bool = False, is_output: bool = False):
        self.name = name
        self.size_bytes = size_bytes
        self.dtype = dtype
        self.is_input = is_input
        self.is_output = is_output
        self.offset = 0
        self.lifetime_start = 0
        self.lifetime_end = 0

    def __repr__(self):
        return f"MemoryRegion(name={self.name}, size={self.size_bytes}, dtype={self.dtype}, offset={self.offset})"


class MemoryPlan:
    def __init__(self, total_size_bytes: int):
        self.total_size_bytes = total_size_bytes
        self.regions: Dict[str, MemoryRegion] = {}
        self.alloc_order: List[str] = []
        self.stream_allocations: Dict[int, List[str]] = {}

    def add_region(self, region: MemoryRegion):
        self.regions[region.name] = region
        self.alloc_order.append(region.name)

    def get_region(self, name: str) -> Optional[MemoryRegion]:
        return self.regions.get(name)


class MemoryPlanner:
    def __init__(self, hardware: HardwareSpec):
        self.hardware = hardware
        self.buffer_reuse_enabled = True
        self.multi_stream_enabled = True

    def plan(self, graph: Graph, kernel_instances: List[KernelInstance]) -> MemoryPlan:
        regions = self._analyze_lifetimes(graph, kernel_instances)
        if self.buffer_reuse_enabled:
            regions = self._apply_buffer_reuse(regions, graph)
        
        plan = self._allocate_memory(regions)
        
        if self.multi_stream_enabled:
            self._assign_streams(plan, kernel_instances)
        
        return plan

    def _analyze_lifetimes(self, graph: Graph, kernel_instances: List[KernelInstance]) -> List[MemoryRegion]:
        regions = []
        node_idx = 0
        
        for node in graph.nodes:
            for out in node.outputs:
                size_bytes = self._estimate_tensor_size(out, graph)
                region = MemoryRegion(name=out, size_bytes=size_bytes, dtype="fp32")
                region.lifetime_start = node_idx
                region.lifetime_end = node_idx + 1
                regions.append(region)
            
            for inp in node.inputs:
                if inp not in [r.name for r in regions]:
                    if graph.get_tensor(inp):
                        size_bytes = self._estimate_tensor_size(inp, graph)
                        region = MemoryRegion(name=inp, size_bytes=size_bytes, dtype="fp32", is_input=True)
                        region.lifetime_start = 0
                        region.lifetime_end = len(graph.nodes)
                        regions.append(region)
            
            node_idx += 1
        
        for region in regions:
            if region.name in graph.outputs:
                region.is_output = True
                region.lifetime_end = len(graph.nodes)
        
        return regions

    def _estimate_tensor_size(self, tensor_name: str, graph: Graph) -> int:
        tensor = graph.get_tensor(tensor_name)
        if tensor and tensor.shape:
            size = 1
            for dim in tensor.shape:
                if isinstance(dim, int):
                    size *= dim
                else:
                    size *= 1
            return size * 4
        return 1024 * 1024

    def _apply_buffer_reuse(self, regions: List[MemoryRegion], graph: Graph) -> List[MemoryRegion]:
        sorted_regions = sorted(regions, key=lambda r: (r.lifetime_end, r.lifetime_start))
        reused = {}
        
        for region in sorted_regions:
            if region.is_input or region.is_output:
                continue
            
            for candidate in sorted_regions:
                if candidate.name == region.name:
                    continue
                if candidate.name in reused:
                    continue
                if not candidate.is_input and not candidate.is_output:
                    if candidate.lifetime_end <= region.lifetime_start:
                        if candidate.size_bytes >= region.size_bytes:
                            reused[region.name] = candidate.name
                            break
        
        merged = {}
        for region in regions:
            if region.name in reused:
                original = merged.get(reused[region.name])
                if original:
                    original.lifetime_end = max(original.lifetime_end, region.lifetime_end)
                else:
                    for r in regions:
                        if r.name == reused[region.name]:
                            r.lifetime_end = max(r.lifetime_end, region.lifetime_end)
                            merged[r.name] = r
            else:
                merged[region.name] = region
        
        return list(merged.values())

    def _allocate_memory(self, regions: List[MemoryRegion]) -> MemoryPlan:
        sorted_regions = sorted(regions, key=lambda r: r.size_bytes, reverse=True)
        
        total_size = sum(r.size_bytes for r in sorted_regions)
        plan = MemoryPlan(total_size_bytes=total_size)
        
        offset = 0
        for region in sorted_regions:
            region.offset = offset
            plan.add_region(region)
            offset += region.size_bytes
        
        return plan

    def _assign_streams(self, plan: MemoryPlan, kernel_instances: List[KernelInstance]):
        stream_count = min(self.hardware.num_sm // 2, 4)
        for i, kernel in enumerate(kernel_instances):
            stream_id = i % stream_count
            if stream_id not in plan.stream_allocations:
                plan.stream_allocations[stream_id] = []
            for out_name in kernel.output_names:
                if out_name in plan.regions:
                    plan.stream_allocations[stream_id].append(out_name)

    def optimize_bandwidth(self, plan: MemoryPlan, kernel_instances: List[KernelInstance]) -> MemoryPlan:
        for kernel in kernel_instances:
            for inp_name in kernel.input_names:
                if inp_name in plan.regions:
                    region = plan.regions[inp_name]
                    region.offset = (region.offset // 128) * 128
        
        return plan


class UnifiedMemoryPool:
    def __init__(self, hardware: HardwareSpec):
        self.hardware = hardware
        self.total_memory = hardware.global_memory_bytes
        self.used_memory = 0
        self.pool: Dict[str, MemoryRegion] = {}

    def allocate(self, name: str, size_bytes: int, dtype: str = "fp32") -> MemoryRegion:
        if name in self.pool:
            return self.pool[name]
        
        region = MemoryRegion(name=name, size_bytes=size_bytes, dtype=dtype)
        region.offset = self.used_memory
        self.pool[name] = region
        self.used_memory += size_bytes
        
        if self.used_memory > self.total_memory:
            raise MemoryError("Out of GPU memory")
        
        return region

    def free(self, name: str):
        if name in self.pool:
            del self.pool[name]

    def get_usage(self) -> Dict:
        return {
            "used_bytes": self.used_memory,
            "total_bytes": self.total_memory,
            "percent_used": (self.used_memory / self.total_memory) * 100,
            "num_allocations": len(self.pool)
        }
