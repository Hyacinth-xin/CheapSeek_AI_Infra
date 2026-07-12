# C1：AEC IR 编译器 - 赛题说明

## 赛题描述

设计一个面向 GPGPU 的编译器，将 PTX 风格中间表示编译为 AEC ISA 机器码，支持指令调度、寄存器分配和多精度 GEMM 优化等编译 pass。请从赛题B中获得ISA的spec。

## 1. 输入

评测系统提供 PTX 风格 IR 文件：

```text
input.ptx
```

编译器必须支持统一的命令行接口：

```bash
aec-cc input.ptx -o output.aecbin
```

可选优化级别：

```bash
aec-cc input.ptx -O0 -o output.aecbin
aec-cc input.ptx -O2 -o output.aecbin
aec-cc input.ptx -O3 -o output.aecbin
```

参赛者可自行设计内部 IR、SSA、CFG 和 pass 流水线，但不得要求修改官方测试输入。

## 2. 输出

编译器必须生成合法的 AEC 二进制：

```text
output.aecbin
```

二进制必须至少包含：

```text
Header
Code Section
Data Section
Relocation Section
Symbol Table
```

Code Section 由 128-bit 定长 AEC ISA 指令组成。

### 反汇编器

参赛者还必须提供反汇编工具：

```bash
aec-objdump output.aecbin
```

输出人类可读的 AEC 汇编。

## 3. IR 格式

PTX 风格 IR 遵循以下通用结构：

```text
.version 1.0
.target aec_sm_10
.kernel vector_add(
  .param .u64 param_a,
  .param .u64 param_b,
  .param .u64 param_c
)
{
  .reg .f32   %f<10>;
  .reg .u32   %r<10>;
  .reg .u64   %rd<10>;
  .reg .pred  %p<5>;

  ld.param.u64    %rd1, [param_a];
  ld.param.u64    %rd2, [param_b];
  ld.param.u64    %rd3, [param_c];
  ...
  add.f32   %f9, %f7, %f8;
  st.gmem.f32   [%rd9], %f9;
  ret;
}
```

## 4. 必须的编译 Pass

### 4.1 IR 解析

- 支持上述 PTX 风格 IR 语法
- IR 数据结构：基本块 + CFG + SSA 形式

### 4.2 优化 Pass

标准优化：
- 常量传播
- 死代码消除（DCE）
- 公共子表达式消除（CSE）
- 循环不变量外提（LICM）
- 基本块合并

GPGPU 特有优化：
- 内存合并访问
- 谓词执行优化
- Shared Memory 缓存提升

多精度 GEMM 优化：
- 自动矩阵乘法模式检测
- 最优精度选择（FP4/FP8/FP16/BF16/FP32/FP64/INT4/INT8/INT32）
- Tile 大小自动调整

### 4.3 寄存器分配

- Linear scan 或图着色寄存器分配
- Spill 代码生成
- 约束：每线程最多 256 个寄存器

### 4.4 指令调度

- 依赖感知调度（DDG 构建 + List Scheduling）
- 延迟隐藏：交织内存和计算指令
- AEC ISA 双发射配对优化

### 4.5 代码生成

- 目标代码生成：128-bit 定长 AEC 指令编码
- 二进制文件格式：Header + Code Section + Data Section + Relocation + Symbol Table

## 5. 评测流程

```text
PTX 测试
   |
   v
参赛者编译器
   |
   v
AEC 二进制
   |
   v
二进制验证
   |
   v
AEC Golden Model
   |
   v
正确性检查
   |
   v
AEC Cycle Model
   |
   v
性能指标
```

正确性是门禁：仅编译结果通过正确性验证的测试才纳入性能评分。主要性能指标为 AEC Cycle Model 的 `total_cycles`。

## 6. 公开测试用例

[testcases/](testcases/) 中 5 道代表性 PTX 题：

| 编号 | 文件 | 类别 | 主题 | 输入规模 |
|------|------|------|------|----------|
| PTX-01 | `PTX-01_vector_add.ptx` | T1 基础 Lowering | Vector Add, FP32 | N=4096, blockDim=256, gridDim=16 |
| PTX-02 | `PTX-02_invariant_poly.ptx` | T2 控制与标量 | Loop Invariant, CSE, DCE | N=256, blockDim=256, gridDim=1 |
| PTX-03 | `PTX-03_repeated_reuse.ptx` | T3 内存 | Load Reuse, Shared Memory | N=4096, blockDim=256, gridDim=16 |
| PTX-04 | `PTX-04_reg_schedule.ptx` | T4 寄存器与调度 | Live Interval, DDG, Dual Issue | N=8192, blockDim=256, gridDim=32 |
| PTX-05 | `PTX-05_gemm_f16.ptx` | T5 Tensor / GEMM | TMUL Lowering, FP16 Tiling | M=N=K=128, blockDim=16, gridDim=8x8 |

## 7. 环境

- 8 CPU 核心
- 16 GB 内存
- 180 秒编译超时
- Docker 运行环境
- 无网络访问
