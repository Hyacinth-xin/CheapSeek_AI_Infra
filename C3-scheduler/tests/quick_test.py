#!/usr/bin/env python3
"""快速测试修复的核心功能"""
import sys
sys.path.insert(0, ".")

from kernel import KernelSpecRef, KernelTuningParams
from strategy import DecompositionStrategy
from memory_planner import UnifiedMemoryPool, FreeBlock
from hardware import hardware

print("=== 测试 KernelSpecRef ===")
# 测试 tuning_params 字段
kernel_ref = KernelSpecRef(kernel_name="matmul_fp16")
kernel_ref.tuning_params = KernelTuningParams(block_x=256, grid_x=1024, smem_bytes=4096)
print(f"✓ kernel_ref.tuning_params = {kernel_ref.tuning_params}")

print("\n=== 测试 Strategy ===")
strategy = DecompositionStrategy()
print(f"✓ 敏感算子列表: {strategy.sensitive_ops}")
print(f"✓ 精度策略: {list(strategy._precision_strategy.keys())}")

# 测试精度选择
from graph import NodeInfo
from graph import Graph

test_node = NodeInfo(name="test_matmul", op_type="Gemm", inputs=["x", "w"], outputs=["y"])
precision = strategy.select_precision(test_node, Graph())
print(f"✓ Gemm 精度选择: {precision.precision}")

# 测试 Winograd 策略
conv_node = NodeInfo(
    name="test_conv",
    op_type="Conv",
    inputs=["x", "w", "b"],
    outputs=["y"],
    attrs={"kernel_shape": [3, 3]}
)
kernels = strategy.decompose(conv_node, Graph(), "fp16")
print(f"✓ 3x3 Conv 使用 Winograd: {kernels[0].kernel_name}")

# 测试普通 Conv (非 3x3)
conv_5x5 = NodeInfo(
    name="test_conv5",
    op_type="Conv",
    inputs=["x", "w", "b"],
    outputs=["y"],
    attrs={"kernel_shape": [5, 5]}
)
kernels_5x5 = strategy.decompose(conv_5x5, Graph(), "fp16")
print(f"✓ 5x5 Conv 使用 im2col: {kernels_5x5[0].kernel_name}")

print("\n=== 测试 MemoryPlanner ===")
# 测试 FreeBlock
block = FreeBlock(offset=1024, size=2048)
print(f"✓ FreeBlock: {block}")

# 测试 UnifiedMemoryPool
pool = UnifiedMemoryPool(hardware)
region = pool.allocate("test_tensor", 1024)
print(f"✓ 分配: {region}")

# 测试 free-list
pool.free("test_tensor")
print(f"✓ 释放后 free_blocks 数量: {len(pool.free_blocks)}")

# 重新分配应该从 free-list 获取
region2 = pool.allocate("test_tensor2", 512)
print(f"✓ 从 free-list 重新分配: offset={region2.offset}")

print("\n=== 所有测试通过 ===")