# C1 编译器公开测试集

本目录包含 C1 编译器赛道的 5 道公开 PTX 测试题，分别覆盖 T1–T5 五个测试类别。
完整评分细则见 `../C1_Compiler_Evaluation_and_Scoring_Spec_CN.md`。

## 测试用例清单

| 编号 | 文件 | 类别 | 主题 | 性能分 |
|------|------|------|------|--------|
| PTX-01 | `PTX-01_vector_add.ptx` | T1 Basic Lowering | Vector Add, FP32 | 无 |
| PTX-02 | `PTX-02_invariant_poly.ptx` | T2 Control & Scalar Opt | Loop Invariant, CSE, DCE | 有 |
| PTX-03 | `PTX-03_repeated_reuse.ptx` | T3 Memory Opt | Load Reuse, Shared Memory Promotion | 有 |
| PTX-04 | `PTX-04_reg_schedule.ptx` | T4 Register & Scheduling | Live Interval, DDG, Dual Issue | 有 |
| PTX-05 | `PTX-05_gemm_f16.ptx` | T5 Tensor / GEMM | TMUL Lowering, Tiling | 有 |

## 公开输入规模

| 编号 | N / Shape | blockDim | gridDim |
|------|-----------|----------|---------|
| PTX-01 | N=4096 | 256 | 16 |
| PTX-02 | N=256 | 256 | 1 |
| PTX-03 | N=4096 | 256 | 16 |
| PTX-04 | N=8192 | 256 | 32 |
| PTX-05 | M=N=K=128 (FP16) | 16 | 8×8 |

## 编译调用方式

```bash
aec-cc PTX-01_vector_add.ptx -O2 -o PTX-01.aecbin
aec-objdump PTX-01.aecbin
```

## 与隐藏测试的关系

公开测试仅用于说明能力范围。隐藏测试可能进行：

- 参数变化（N、矩阵规模）
- 寄存器重命名
- 基本块重排
- 循环次数变化
- 插入 Dead Code
- 增加 Register Pressure
- 改变数据类型（PTX-05 的 FP16 → FP4/FP8/BF16/FP32/INT4/INT8/INT32）
- 改变 Memory Reuse Pattern

参赛队伍不得假设公开题结构固定。

## PTX-05 隐藏测试精度矩阵

隐藏测试将覆盖 AEC ISA 支持的全部 9 种 GEMM 精度：

- FP4 E2M1
- FP8 E4M3
- FP8 E5M2
- FP16
- BF16
- FP32
- INT4
- INT8
- INT32

以及多种矩阵规模：

- 64 × 64 × 64
- 128 × 128 × 128
- 256 × 128 × 512
- 512 × 512 × 256
- 非 16 倍数边界规模
