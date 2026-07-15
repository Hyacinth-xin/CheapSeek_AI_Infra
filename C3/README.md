# C3 — Operator Scheduling & Model Deployment

## 框架说明

本模块为 AEC 编译器提供算子调度策略与端到端模型推理引擎，全部框架源码位于 `src/` 下。

| 子模块 | 文件                          | 功能                                    |
| ------ | ----------------------------- | --------------------------------------- |
| C3.1   | `src/export_dag.py`, `src/graph.py` | ONNX 模型解析 → DAG JSON 导出             |
| C3.2   | `src/strategy.py`             | 多精度路由 + Winograd/im2col 策略分解    |
| C3.3   | `src/graph_passes/fusion.py`  | 5 种算子融合 pass（FusedMatMulBias 等）  |
| C3.4   | `src/memory_planner.py`, `src/hardware.py`, `src/kernel.py` | FreeBlock 内存规划 + 权重预加载 + 流并行  |
| C3.5   | `src/backend.py`, `src/executor.py`, `src/data_loader.py`, `src/data_writer.py`, `src/infer.py`, `src/infer_worker.py` | numpy/cupy 双后端推理引擎，支持 18 种 ONNX 算子 |

## 文件结构

```
C3/
├── README.md
└── src/                          # 框架源码
    ├── __init__.py
    ├── export_dag.py             # C3.1 CLI 入口
    ├── graph.py                  # DAG 图数据结构
    ├── hardware.py               # GPU 硬件规格定义 (128SM / 48GB)
    ├── kernel.py                 # KernelSpec / KernelInstance 定义
    ├── memory_planner.py         # C3.4 内存规划 (FreeBlock + 权重预加载 + 流并行)
    ├── strategy.py               # C3.2 多精度策略 + 算子分解
    ├── infer.py                  # C3.5 CLI 推理入口
    ├── infer_worker.py           # C3.5 Worker 入口（持久进程，stdin/stdout 协议）
    ├── backend.py                # numpy/cupy 双后端兼容层
    ├── executor.py               # ONNX 图解释执行器 (18 种算子)
    ├── data_loader.py            # 输入数据加载 (manifest.json + .npy)
    ├── data_writer.py            # 输出数据写盘 (manifest.json + .npy)
    └── graph_passes/
        ├── __init__.py
        └── fusion.py             # C3.3 算子融合 (5 种 pattern)
```

## C3.1 — DAG 导出

从 ONNX 模型导出 DAG JSON：

```
python src/export_dag.py --onnx <model.onnx> --output <dag.json>
```

示例：

```
python src/export_dag.py --onnx models/mlp_v1.onnx --output dag_mlp.json
```

## C3.5 — 模型推理

### CLI 模式（单次推理）

```
python src/infer.py --onnx <model.onnx> --input <input_dir> --output <output_dir> --batch-size 256
```

示例：

```
python src/infer.py --onnx models/mlp_v1.onnx --input testdata/c35/mlp_v1/input --output output/ --batch-size 256
```

### Worker 模式（持久进程，评测通道）

启动 Worker：

```
python src/infer_worker.py
```

Worker 遵循 `C35_WORKER_PROTOCOL.md` 协议：
- 启动后输出 `READY`
- 通过 stdin 接收 JSON 任务（每行一个），stdout 逐行输出 JSON 结果
- 收到 `{"cmd": "exit"}` 时退出

任务格式：

```json
{
    "onnx": "models/mlp_v1.onnx",
    "input": "testdata/c35/mlp_v1/input",
    "output": "output/mlp_v1",
    "batch_size": 256
}
```

## 后端说明

- **numpy 后端**（默认）：本地无 GPU 时自动回退，用于逻辑正确性验证
- **cupy 后端**：检测到 cuda-cupy 时自动启用，用于服务器 GPU 推理与 NVML 显存采样
