# C1 编译器公开测试集

本目录包含 C1 编译器赛道的 5 道公开 PTX 测试题，分别覆盖 T1–T5 五个测试类别。

完整评分细则见 `../docs/scoring.md`。

## 目录结构

每个测试用例由 `kernel.ptx` 和 `manifest.json` 组成：

```text
T1_basic_lowering/
  kernel.ptx
  manifest.json
T2_scalar_optimization/
  kernel.ptx
  manifest.json
T3_memory_reuse/
  kernel.ptx
  manifest.json
T4_register_scheduling/
  kernel.ptx
  manifest.json
T5_scalar_gemm/
  kernel.ptx
  manifest.json
```

## 测试用例清单

| 编号 | 目录 | 类别 | 主题 | 性能分 |
|------|------|------|------|--------|
| T1 | `T1_basic_lowering/` | T1 基础 Lowering | Vector Add, FP32 | 无 |
| T2 | `T2_scalar_optimization/` | T2 标量优化 | Repeated Expression, CSE, DCE | 有 |
| T3 | `T3_memory_reuse/` | T3 内存访问优化 | Load Reuse, 地址计算 | 有 |
| T4 | `T4_register_scheduling/` | T4 寄存器与调度 | Live Interval, DDG, 调度 | 有 |
| T5 | `T5_scalar_gemm/` | T5 FP32 Scalar GEMM | 标量 GEMM, K维循环 | 有 |

## 编译调用方式

```bash
compiler/aec-cc kernel.ptx -O2 -o output.aecbin --report compile_report.json
```

## 与隐藏测试的关系

公开测试仅用于说明能力范围。隐藏测试可能进行：

- 参数变化（N、矩阵大小）
- 寄存器重命名
- 基本块重排
- 循环次数变化
- 插入死代码
- 增加寄存器压力
- 地址计算形式变化
- 内存访问模式变化
- 标量 GEMM 矩阵大小和边界变化

参赛队伍不得假设公开题结构固定。
