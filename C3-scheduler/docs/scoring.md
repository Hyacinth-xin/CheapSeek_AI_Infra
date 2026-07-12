# C3：算子调度与模型部署 - 评分细则

## 总分：100 分

| 子任务 | 分值 | 评测方式 |
|--------|-----:|----------|
| C3.1 计算图解析与表示 | 10 | 自动检查 |
| C3.2 算子分解与内核选择 | 15 | 微基准测试 |
| C3.3 算子融合与图优化 | 15 | 微基准测试 |
| C3.4 内存规划与调度 | 10 | Code Review |
| C3.5 典型模型部署 | 50 | 端到端测试 |
| **合计** | **100** | |

---

## C3.1 计算图解析（10 分）

- 完成模型加载：4 分
- 正确的计算图解析：6 分

---

## C3.2 算子分解（15 分）

评测脚本：`benchmarks/c32_c33/bench_c32_c33.py`
评测模型：MNIST MLP、CIFAR-10 简化 ResNet-18

| 维度 | 分值 |
|------|-----:|
| D1. 多精度路由正确性 / 覆盖度 | 3 |
| D2. 内核序列完整性 | 3 |
| D3. 中间张量跟踪 | 3 |
| D4. 内核调优参数有效性 | 3 |
| D5. 硬件能力覆盖度 | 3 |

### D1. 多精度路由正确性 / 覆盖度（3 分）

| 子项 | 满分 | 检查内容 |
|------|-----:|----------|
| 敏感算子强制 FP32 | 1.5 | Softmax / LayerNorm / BatchNorm / ReduceMax / ReduceSum / ReduceMean 的 `precision == "fp32"` 占比 × 1.5 |
| 精度多样度 | 1.0 | 出现 fp32 / fp16 / fp8 / fp4 中的 N 种，得分 = `N / 4` |
| 非敏感算子走可用精度 | 0.5 | MatMul / Linear / Conv2d 的精度 ∈ `hardware.supported_precisions()` 占比 × 0.5 |

> 硬指标：FULL_FP32 模式下 `max_abs_diff ≤ 1e-3`、`top1_match ≥ 0.99`。强行对敏感算子开低精度导致超阈，直接扣光 D1。

### D2. 内核序列完整性（3 分）

`score = seq_coverage × 1.0 + key_seq_score × 2.0`，最高 3.0。

关键 kernel 检查：MatMul(`matmul_*`)、Softmax(`reduce_max`+`exp`+`reduce_sum`+`div`)、LayerNorm(`reduce_mean`+`sub`+`mul`+`sqrt`)、Conv2d(`winograd_forward_*` 或 `im2col_*`)。

### D3. 中间张量跟踪（3 分）

`score = key_intermediate_ratio × 2.0 + total_intermediate_ratio × 1.0`，最高 3.0。

### D4. 内核调优参数有效性（3 分）

| 子项 | 满分 | 检查内容 |
|------|-----:|----------|
| `tuning_coverage` | 1.5 | 产出非空 `tuning_params` 的算子占比（目标 ≥ 90%） |
| `tuning_validity` | 1.5 | 3 条断言：① `0 < block_x ≤ max_threads_per_block`，② `grid_x > 0`，③ `smem_bytes ≤ hardware.smem_bytes` |

### D5. 硬件能力覆盖度（3 分）

| 子项 | 满分 | 检查内容 |
|------|-----:|----------|
| 精度种类 | 1.0 | ≥ 2 种得 0.5，3–4 种满分 |
| GEMM kernel 多样度 | 1.0 | 至少 `matmul_f32` + `matmul_f16`，再出现 `matmul_f8`/`matmul_f4` 各 +0.25 |
| Conv2d 策略选择 | 1.0 | im2col 与 Winograd 都被选过 |

---

## C3.3 算子融合（15 分）

| 维度 | 分值 |
|------|-----:|
| F1. 融合 pattern 覆盖 | 5 |
| F2. Kernel launch 数减少 | 3 |
| F3. 中间 buffer 数减少 | 3 |
| F4. 融合正确性 | 4 |

### F1. 融合 pattern 覆盖（5 分）

5 个目标 pattern，命中 1 个 +1 分：

| Pattern | 触发条件 |
|---------|----------|
| `FusedMatMulBias` | MatMul → AddBias |
| `FusedConv2dBatchNorm` | Conv2d → BatchNorm |
| `FusedEWChain` | 2–5 个相邻 elementwise |
| `FusedSoftmaxDropout` | Softmax → Dropout |
| `FusedResidualNorm` | skip-Add → LayerNorm |

> 当前 ResNet-18 ONNX 无 BN 节点（训练时已折进 conv 权重），需写预融合 pass 才能命中 FusedConv2dBN。

### F2. Kernel launch 数减少（3 分）

`score = min((raw_launches − opt_launches) / raw_launches × 5.0, 3.0)`，reduction ≥ 60% 即满分。

### F3. 中间 buffer 数减少（3 分）

`score = min((raw_buffers − opt_buffers) / raw_buffers × 5.0, 3.0)`，reduction ≥ 60% 即满分。

### F4. 融合正确性（4 分）

| 检查项 | 分值 |
|--------|-----:|
| `graph.outputs` 保留可解析 | 1 |
| `graph.inputs` 保留 | 1 |
| `graph.validate()` 通过 | 1 |
| 优化图节点数 ≤ 原始图节点数 | 1 |

外加数值对齐检查：`MockRuntime` 跑原始图 + 优化图，FP32 参考 `max_abs_diff ≤ 1e-3`；任一超阈则 F4 全扣。

---

## C3.4 内存规划（10 分）

Code Review 评测，五项各 2 分：

| 子项 | 审查要点 | 满分条件 |
|------|----------|----------|
| A. 设备内存池与权重预加载 | 设备内存分配/释放接口 + 权重经计划上传到 device buffer | 完整链路 |
| B. 中间张量 lifetime 复用 | 生命周期不重叠的张量映射到同一 slot / 物理缓冲 | 接入执行计划 |
| C. 内存池碎片整理 | free 后块进入可复用结构 + best-fit / size class / coalesce | 块管理策略 |
| D. 权重预取 | 部分层权重的 alloc/h2d 前移到前序层计算附近 | 边算边传语义 |
| E. 流级并行 | 无数据依赖的算子分配到不同 compute stream | 多 stream 计划 |

等级：

| 等级 | 分数 | 含义 |
|------|------|------|
| 未通过 | 0–3 | 仅有零散封装 |
| 基础 | 4–5 | 至少两条主线满分 |
| 良好 | 6–7 | 内存侧与重叠侧均有完整实现 |
| 优秀 | 8–10 | 五项多数满分，链路可闭环追溯 |

---

## C3.5 端到端部署（50 分）

### 分值分配

| 维度 | 分值 | 性质 |
|------|-----:|------|
| 精度测试 + 准确率 | 15 | 通过门槛 |
| 运行时间效率 | 25 | 排序加分 |
| 峰值显存占用 | 10 | 排序加分 |

### 门禁检查

- **精度门禁**：`numpy.allclose(out, golden, rtol=1e-3, atol=1e-3)`
- **准确率门禁**：MLP（MNIST）≥ 98%，ResNet-18（CIFAR-10）≥ 85%

未通过门禁的模型得 0 分。

### 运行时间（25 分）

评测机记录选手程序从启动到退出的时间。得分根据所有提交中的运行时间排名计算。

### 峰值显存（10 分）

评测机通过 NVML 按进程（含子进程）采样 GPU 已用显存并取峰值。得分根据排名计算。

---

## 评测门槛参考（C3.2 + C3.3）

| 总分区间 | 评语 |
|----------|------|
| ≥ 25 | S 级 |
| 20 – 24 | A 级 |
| 14 – 19 | B 级 |
| 8 – 13 | C 级 |
| < 8 | 未达标 |

## 评测说明

- ONNX 模型导出时已将 BN 折叠到 Conv 权重中（图中无 BN 节点）
- 精度阈值 `rtol = atol = 1e-3` 适用于所有模型
- 命令模板必须在报名时提交，包含 `{onnx}`、`{input}`、`{output}` 占位符
- 三个模型的批量维 `N` 均为动态维，支持任意批量大小
- 输入数据已完成预处理，选手程序无须再做任何预处理
