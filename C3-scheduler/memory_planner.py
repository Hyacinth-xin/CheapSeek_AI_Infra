from typing import Dict, List, Tuple, Optional, Set
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
        """基于依赖分析的流级并行调度"""
        stream_count = min(self.hardware.num_sm // 2, 4)
        dependency_graph = self._build_dependency_graph(kernel_instances)

        # 使用拓扑排序 + 流分配
        assigned_streams: Dict[int, List[int]] = {}  # stream_id -> kernel_indices
        kernel_stream_assign: List[int] = [-1] * len(kernel_instances)

        for i in range(len(kernel_instances)):
            # 找到所有已经完成的流
            ready_streams = []
            for sid in range(stream_count):
                # 检查这个流上的最后一个 kernel 是否已完成（无未完成的依赖）
                if sid not in assigned_streams:
                    ready_streams.append(sid)
                    continue

                last_kernel_idx = assigned_streams[sid][-1]
                # 检查是否有 kernel 依赖于这个流上的输出
                has_dependent = any(i in dependency_graph.get(last_kernel_idx, []) for i in range(len(kernel_instances)))
                if not has_dependent and kernel_stream_assign[i] == -1:
                    ready_streams.append(sid)

            # 选择第一个可用的流（简单的轮询）
            if ready_streams:
                stream_id = ready_streams[0]
            else:
                stream_id = 0

            if stream_id not in assigned_streams:
                assigned_streams[stream_id] = []

            assigned_streams[stream_id].append(i)
            kernel_stream_assign[i] = stream_id

            # 记录分配
            if stream_id not in plan.stream_allocations:
                plan.stream_allocations[stream_id] = []
            for out_name in kernel_instances[i].output_names:
                if out_name in plan.regions:
                    plan.stream_allocations[stream_id].append(out_name)

    def _build_dependency_graph(self, kernel_instances: List[KernelInstance]) -> Dict[int, Set[int]]:
        """构建 kernel 间的依赖图"""
        # 朴素实现：基于输入/输出名称构建依赖
        output_map: Dict[str, int] = {}  # output_name -> kernel_index

        for i, kernel in enumerate(kernel_instances):
            for out_name in kernel.output_names:
                output_map[out_name] = i

        dependency_graph: Dict[int, Set[int]] = {}

        for i, kernel in enumerate(kernel_instances):
            deps = set()
            for inp_name in kernel.input_names:
                if inp_name in output_map:
                    producer_idx = output_map[inp_name]
                    if producer_idx != i:
                        deps.add(producer_idx)
            dependency_graph[i] = deps

        return dependency_graph

    def optimize_bandwidth(self, plan: MemoryPlan, kernel_instances: List[KernelInstance]) -> MemoryPlan:
        for kernel in kernel_instances:
            for inp_name in kernel.input_names:
                if inp_name in plan.regions:
                    region = plan.regions[inp_name]
                    region.offset = (region.offset // 128) * 128
        
        return plan


class FreeBlock:
    """空闲内存块，用于碎片整理"""
    def __init__(self, offset: int, size: int):
        self.offset = offset
        self.size = size

    def __repr__(self):
        return f"FreeBlock(offset={self.offset}, size={self.size})"


class UnifiedMemoryPool:
    """设备内存池，支持碎片整理和权重预加载"""
    def __init__(self, hardware: HardwareSpec):
        self.hardware = hardware
        self.total_memory = hardware.global_memory_bytes
        self.used_memory = 0
        self.pool: Dict[str, MemoryRegion] = {}
        # Free-list 用于碎片整理
        self.free_blocks: List[FreeBlock] = []
        # 权重预加载队列
        self.prefetch_queue: List[Tuple[str, int]] = []
        # H2D 异步传输标记
        self.async_transfers: Dict[str, bool] = {}

    def allocate(self, name: str, size_bytes: int, dtype: str = "fp32") -> MemoryRegion:
        # 尝试从 free-list 中找到合适的块
        allocated_block = self._find_and_remove_free_block(size_bytes)
        if allocated_block:
            region = MemoryRegion(name=name, size_bytes=size_bytes, dtype=dtype)
            region.offset = allocated_block.offset
            self.pool[name] = region
            return region

        # 否则从末尾分配
        region = MemoryRegion(name=name, size_bytes=size_bytes, dtype=dtype)
        region.offset = self.used_memory
        self.pool[name] = region
        self.used_memory += size_bytes

        if self.used_memory > self.total_memory:
            raise MemoryError("Out of GPU memory")

        return region

    def _find_and_remove_free_block(self, size: int) -> Optional[FreeBlock]:
        """从 free-list 中找到最合适的块并移除（best-fit）"""
        suitable = [(i, b) for i, b in enumerate(self.free_blocks) if b.size >= size]
        if not suitable:
            return None
        # Best-fit: 找到最小的足够大的块
        suitable.sort(key=lambda x: x[1].size)
        idx, best = suitable[0]
        if best.size > size:
            # 如果块太大，分割它
            self.free_blocks[idx] = FreeBlock(best.offset + size, best.size - size)
            return FreeBlock(best.offset, size)
        else:
            # 完全匹配，移除块
            del self.free_blocks[idx]
            return best

    def free(self, name: str):
        if name not in self.pool:
            return

        region = self.pool[name]
        # 添加到 free-list
        self.free_blocks.append(FreeBlock(region.offset, region.size_bytes))
        del self.pool[name]

        # 合并相邻的空闲块（coalesce）
        self._coalesce_free_blocks()

    def _coalesce_free_blocks(self):
        """合并相邻的空闲块，减少碎片"""
        if len(self.free_blocks) < 2:
            return

        # 按 offset 排序
        self.free_blocks.sort(key=lambda b: b.offset)

        merged = []
        for block in self.free_blocks:
            if merged and merged[-1].offset + merged[-1].size == block.offset:
                # 合并相邻块
                merged[-1] = FreeBlock(merged[-1].offset, merged[-1].size + block.size)
            else:
                merged.append(block)

        self.free_blocks = merged

    def upload_weight(self, name: str, size_bytes: int, async_transfer: bool = False) -> MemoryRegion:
        """权重预加载到设备内存"""
        region = self.allocate(name, size_bytes, dtype="fp32")
        if async_transfer:
            self.async_transfers[name] = True
        return region

    def prefetch_weights(self, weights: List[Tuple[str, int]], offset_offset: int = 0):
        """预取下一层的权重"""
        self.prefetch_queue = weights
        for name, size in weights:
            # 标记为异步传输
            self.async_transfers[name] = True

    def get_usage(self) -> Dict:
        free_size = sum(b.size for b in self.free_blocks)
        return {
            "used_bytes": self.used_memory,
            "total_bytes": self.total_memory,
            "free_bytes": free_size,
            "percent_used": (self.used_memory / self.total_memory) * 100,
            "num_allocations": len(self.pool),
            "num_free_blocks": len(self.free_blocks),
            "fragmentation": self._calculate_fragmentation()
        }

    def _calculate_fragmentation(self) -> float:
        """计算内存碎片率"""
        if not self.free_blocks or sum(b.size for b in self.free_blocks) == 0:
            return 0.0
        max_free = max(b.size for b in self.free_blocks)
        total_free = sum(b.size for b in self.free_blocks)
        # 碎片率 = 1 - 最大空闲块 / 总空闲块
        return 1.0 - (max_free / total_free)
