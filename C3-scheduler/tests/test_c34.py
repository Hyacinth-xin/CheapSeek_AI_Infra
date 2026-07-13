import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph import import_onnx_graph
from hardware import hardware
from memory_planner import MemoryPlanner, UnifiedMemoryPool
from strategy import strategy
from kernel import KernelInstance

g = import_onnx_graph('models/mlp_v1.onnx')
print('模型:', 'mlp_v1.onnx')
print('节点数:', len(g.nodes))

planner = MemoryPlanner(hardware)
kernel_instances = []
for node in g.nodes:
    pp = strategy.select_precision(node, g)
    kernel_specs = strategy.decompose(node, g, pp.precision)
    for spec in kernel_specs:
        ki = KernelInstance(
            kernel_name=spec.kernel_name,
            input_names=spec.inputs,
            output_names=spec.outputs,
            precision=pp.precision
        )
        kernel_instances.append(ki)

plan = planner.plan(g, kernel_instances)
plan = planner.optimize_bandwidth(plan, kernel_instances)

print('\n内存规划:')
print('总大小:', plan.total_size_bytes, 'bytes')
print('分配区域数:', len(plan.regions))
for name, region in plan.regions.items():
    print('  ', name, ':', region.size_bytes, 'bytes, offset=', region.offset)

print('\n流分配:')
for stream_id, names in plan.stream_allocations.items():
    print('  Stream', stream_id, ':', names)

print('\n统一内存池:')
pool = UnifiedMemoryPool(hardware)
for name, region in plan.regions.items():
    pool.allocate(name, region.size_bytes)

usage = pool.get_usage()
print('  使用:', usage['used_bytes'], '/', usage['total_bytes'], 'bytes (', usage['percent_used'], '%)')
print('  分配数:', usage['num_allocations'])
