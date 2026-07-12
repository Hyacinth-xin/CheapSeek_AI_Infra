# 赛道 C 项目指令文档（供 AI 编码工具使用）

## 项目背景

这是一个 GPGPU 智能体加速设计竞赛的赛道 C 项目。我们需要在 63 小时内（7月12日19:00 - 7月15日10:00）完成三个独立子题：

- **C1**：AEC IR 编译器（PTX → AEC 二进制），满分100
- **C2**：AEC Runtime + 虚拟设备驱动，满分100
- **C3**：算子调度与模型部署（ONNX → 端到端推理），满分100

三个子题独立评分，等级分为 Basic / Good / Excellent。

## 开发环境

- OS: Ubuntu 22.04 (Docker 评测环境)
- 编译器: GCC 13.3.0
- Python: 3.10+
- 构建工具: Make / CMake
- 评测: 无网络访问，Docker 隔离

## 三人分工

| 成员 | 负责子题 | 开发目录 |
|---|---|---|
| A | C1 编译器 | `dev/C1-compiler/` |
| B | C2 Runtime | `dev/C2-runtime/` |
| C | C3 调度器 | `dev/C3-scheduler/` |

---

## C1：AEC IR 编译器

### 任务
将 PTX 风格中间表示编译为 AEC ISA 128-bit 定长机器码。

### 命令行接口
```bash
aec-cc input.ptx -O2 -o output.aecbin    # 编译
aec-objdump output.aecbin                # 反汇编
```

### 编译器流水线
1. 词法分析 → Token 流
2. 语法分析 → AST
3. IR 构建 → SSA 形式基本块 + CFG
4. 优化 pass（必需）：
   - 常量传播、DCE、CSE、LICM、基本块合并
   - 内存合并访问、谓词执行优化、Shared Memory 缓存
   - 多精度 GEMM 模式检测与 Tiling
5. 图着色寄存器分配（256 寄存器上限，Spill 处理）
6. 依赖感知指令调度（DDG + List Scheduling + 双发射配对）
7. 128-bit 指令编码 → AEC 二进制（Header + Code + Data + Relocation + Symbol Table）

### PTX 输入格式示例
```ptx
.version 1.0
.target aec_sm_10
.kernel vector_add(
  .param .u64 param_a,
  .param .u64 param_b,
  .param .u64 param_c
)
{
  .reg .f32   %f<10>;
  .reg .u64   %rd<10>;
  ld.param.u64    %rd1, [param_a];
  add.f32         %f9, %f7, %f8;
  st.gmem.f32     [%rd9], %f9;
  ret;
}
```

### 评分（100分）
- 正确性 50：100 道隐藏测试（T1基础Lowering×20 + T2控制标量×20 + T3内存×20 + T4寄存器调度×20 + T5 GEMM×20），编译错误直接0分
- 性能 35：AEC Cycle Model 的 total_cycles，与基线比较
- 鲁棒性 5：50 道变异测试
- Agent 10：自动选择编译配置的闭环优化（独立运行→读报告→重编译→验证→生成报告）

### 公开测试
5 道代表性 PTX 题位于 `public/Track-C/C1-compiler/testcases/`：
- PTX-01: vector_add (FP32, N=4096)
- PTX-02: invariant_poly (循环不变量, CSE, DCE)
- PTX-03: repeated_reuse (Load Reuse, Shared Memory)
- PTX-04: reg_schedule (Live Interval, DDG, Dual Issue)
- PTX-05: gemm_f16 (TMUL Lowering, FP16 Tiling, M=N=K=128)

### 隐藏测试覆盖
PTX-05 覆盖全部 9 种 GEMM 精度：FP4/FP8(E4M3,E5M2)/FP16/BF16/FP32/FP64/INT4/INT8/INT32
矩阵大小：64³, 128³, 256×128×512, 512²×256, 非16倍数边界

---

## C2：AEC Runtime + 虚拟设备

### 任务
实现 `libaec.so`（Host Runtime C++ 库），与主办方提供的虚拟设备 `libaec_device.so` 交互。

### 不可修改的公共文件（只读）
- `include/aec_runtime.h` — Runtime API 函数签名
- `include/aec_device_abi.h` — 设备层 ABI 接口
- `include/aec_isa.h` — ISA 常量定义
- `lib/libaec_device.so` — 虚拟设备库
- `kernels/images/*.aecbin` — 34 个固定内核镜像
- `kernels/manifest.json` — 内核清单
- `cases/test_r*.py` — 公开测试脚本
- `grader/public_grade.py` — 评分脚本

### 可修改的文件
- `src/aec_runtime.cpp` — 主 Runtime 实现（starter-kit 已给框架）
- `agents/dma_agent.py` — DMA 调度 Agent
- `agents/kernel_agent.py` — 内核选择 Agent
- `Makefile` — 构建脚本

### 虚拟设备参数
- 1 个设备，64 MiB 内存
- 2 个 DMA 通道
- 最大 1024 线程/块
- ABI 版本 2，ISA 版本 2/profile 1
- 参数块最大 64 字节
- 内存分配：64 字节对齐，deterministic lowest-address first-fit

### 16 个 Requirement

| ID | 内容 | 分值 | 等级 |
|---|---|---|---|
| R101 | 设备查询、ISA信息、错误名、TLS last error | 4 | Basic |
| R102 | allocation/free、OOM、reuse、非法free | 6 | Basic |
| R103 | 同步H2D/D2H拷贝、边界检查 | 6 | Basic |
| R104 | Vector Add内核启动、grid/block映射 | 4 | Basic |
| R105 | Stream FIFO、异步操作 | 5 | Good |
| R106 | Event生成、cycles计时、异步错误 | 5 | Good |
| R201 | FP32/INT32 GEMM | 10 | Basic |
| R202 | FP4/FP8/FP16/BF16/FP64 GEMM | 10 | Good |
| R203 | INT4/INT8 + INT32饱和输出 | 4 | Good |
| R204 | FP32 AXPY、DOT、NRM2 | 6 | Good |
| R301 | ABI序列、resolve、completion、stats | 6 | Good |
| R302 | 双DMA通道、异步边界与恢复 | 6 | Good |
| R303 | host registration与zero-copy | 4 | Good |
| R304 | DMA/ISA fault传播与恢复 | 4 | Good |
| R401 | DMA policy Agent | 10 | Excellent |
| R402 | Kernel-image policy Agent | 10 | Excellent |

### 等级门槛
- Basic (30+): R101-R104 + R201 全通过
- Good (75+): + R105-R106 + R202-R204 + R301-R304 全通过
- Excellent (90+): + R401/R402 correctness通过 + 两个Agent都有正的隐藏平均加速比

### Agent 协议
Agent 是独立 Python 脚本，从 stdin 读 JSON，向 stdout 写 JSON。
- 单次超时 1 秒
- 不得访问网络、评分器文件或在 case 间保存状态
- 输出不能带额外字段或日志

### DMA Agent
输入: `{case_id, direction, bytes, alignment, registered, concurrency}`
输出: `{channel(0|1), chunk_bytes(4096|65536|1048576), queue_depth(1|2|4|8), use_zero_copy(bool)}`
DMA虚拟周期 = setup + ceil(ceil(bytes/32)/parallelism) + 24×(ceil(bytes/chunk_bytes)-1) + alignment_penalty
- registered zero-copy 的 setup=45，否则=100
- parallelism=min(queue_depth,concurrency,2)
- 对齐<64时 penalty=13

### Kernel Agent
输入: `{case_id, dtype, m, n, k, alignment, workspace, candidates[]}`
输出: `{kernel_id(string)}`
- naive: 所有合法shape
- tiled: M/N/K均可被4整除
- vectorized: M/N/K均可被8整除，且alignment≥16
- 必须满足workspace和candidate自身约束
- Kernel周期来自实际AEC image interpretation

### 34 个内核镜像
image_id_formula = (semantic_kernel_id << 16) | (dtype << 8) | variant
- vector_add_f32 (1个)
- axpy/dot/nrm2_f32 (3个)
- gemm × 3变体(naive/tiled/vectorized) × 10精度 = 30个
精度映射: 1=FP4, 2=FP8_E4M3, 3=FP8_E5M2, 4=FP16, 5=BF16, 6=FP32, 7=FP64, 8=INT4, 9=INT8, 10=INT32

---

## C3：算子调度与模型部署

### 5 个子任务

| 子任务 | 分值 | 内容 |
|---|---|---|
| C3.1 | 10 | ONNX → DAG JSON |
| C3.2 | 15 | 算子分解（大矩阵拆小kernel） |
| C3.3 | 15 | 算子融合（5种标准+2种自定义） |
| C3.4 | 10 | 内存规划（张量生命周期+重用） |
| C3.5 | 50 | 端到端推理（精度+准确率+时间+显存） |

### 命令行接口
```bash
# C3.1
python export_dag.py --onnx {onnx} --output {output}

# C3.5
python infer.py --onnx {onnx} --input {input} --output {output} --batch-size 256
```

### 3 个模型

| 模型 | 任务 | 输入 | 输出 | 准确率门槛 |
|---|---|---|---|---|
| MLP | MNIST分类 | [N,1,28,28] float32 | [N,10] logits | ≥98% |
| ResNet-18 | CIFAR-10分类 | [N,3,32,32] float32 | [N,10] logits | ≥85% |
| Transformer | 合成序列 | [N,18] int64 | [N,18,14] float32 | 无（精度过即可） |

### C3.5 评分
- 精度门槛: numpy.allclose(out, golden, rtol=1e-3, atol=1e-3) — 使用PyTorch fp32参考输出
- 推理时间: 25分（排序竞争）
- 峰值显存: 15分（排序竞争）
- 精度+准确率不通过 = C3.5直接失败

### 17 种 ONNX 算子（必须全部支持）
Add, Constant, Conv, Div, Erf, Flatten, Gather, Gemm, GlobalAveragePool, LayerNormalization, MatMul, Mul, Relu, Reshape, Softmax, Split, Transpose

### 重要细节
- ResNet的BN已融合进Conv权重，ONNX中无BN节点
- Transformer的GELU分解为 Div+Erf+Add+Mul，无单独Gelu节点
- Gather仅用于词嵌入查表
- 输入数据已预处理，无需再做归一化

### 调试数据
每个公开模型提供: input/(manifest.json+.npy) + golden/(manifest.json+.npy) + labels.npy + thresholds.json

### C3 与 C2 的依赖关系
C3.5 端到端推理需要 C2 的 Runtime API（aecMatmulF32 等）。
如果 C2 未就绪，C3.5 可先用 PyTorch CPU 模拟执行来验证精度。

---

## 开发文件夹结构

```
dev/
├── C1-compiler/           # 成员A的工作目录
│   ├── src/               # 编译器源码
│   ├── testcases/         # 从 public 复制的 5 个 PTX 测试
│   └── agent/             # 编译优化 Agent
│
├── C2-runtime/            # 成员B的工作目录（从 starter-kit 复制）
│   ├── include/           # 公共头文件（只读，不可修改）
│   │   ├── aec_runtime.h
│   │   ├── aec_device_abi.h
│   │   └── aec_isa.h
│   ├── lib/               # 虚拟设备库（只读，不可修改）
│   │   └── libaec_device.so
│   ├── src/               # Runtime 实现（可修改）
│   │   └── aec_runtime.cpp
│   ├── kernels/           # 内核镜像（只读，不可修改）
│   │   ├── manifest.json
│   │   └── images/*.aecbin
│   ├── agents/            # Agent 实现（可修改）
│   │   ├── dma_agent.py
│   │   └── kernel_agent.py
│   ├── cases/             # 公开测试脚本（只读）
│   ├── schemas/           # Agent JSON Schema（只读）
│   ├── golden/            # 参考数据（只读）
│   ├── grader/            # 评分脚本（只读）
│   ├── examples/          # 示例代码（只读）
│   ├── docs/              # 文档（只读）
│   └── Makefile           # 构建脚本（可修改）
│
├── C3-scheduler/          # 成员C的工作目录
│   ├── src/               # 调度器源码
│   ├── models/            # 3 个公开 ONNX 模型
│   ├── testdata/          # 调试数据
│   └── docs/              # 文档
│
└── shared/                # 三人共享
    ├── docs/              # 所有赛题文档
    └── scripts/           # 工具脚本
```

## 关键约束
1. 不得修改公共头文件、设备库、内核镜像、grader
2. 评测环境无网络访问
3. C1 编译超时 180 秒
4. C2 Agent 单次超时 1 秒，stdout+stderr 不超过 64 KiB
5. C3 精度门槛 rtol=atol=1e-3，使用低精度容易超出阈值
6. 每个 Requirement 是 all-or-zero：公共和隐藏检查全部通过才得分
