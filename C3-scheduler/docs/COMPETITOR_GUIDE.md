# C3 赛题

本文档面向参赛选手，规定 C3 赛道各子任务的提交要求、命令行接口、数据格式与自动评测规则。请在开发前完整阅读。

评测采用固定的命令行模板调用选手程序。选手在报名时须提交对应的命令模板，其中以占位符表示评测机在运行时填入的路径。

---

## C3.1

### 1. 任务描述

实现一个命令行程序，读取指定的 ONNX 模型文件，解析其计算图，并将计算图以有向无环图（DAG）的形式导出为一个 JSON 文件。

本任务测试计算图的解析与表示能力。

### 2. 命令行接口

评测机以如下形式调用选手程序：

```
<选手程序> --onnx <model.onnx> --output <dag.json>
```

- `--onnx`：输入 ONNX 模型文件的路径。
- `--output`：输出 DAG JSON 文件的路径，程序须将结果写入该路径。
- 程序须以退出码 0 结束。非零退出码视为该模型处理失败。
- 标准输出（stdout）的内容不参与评测，评测仅读取 `--output` 指定的文件。

报名时须提交命令模板，使用 `{onnx}` 与 `{output}` 作为占位符，例如：

```
python export_dag.py --onnx {onnx} --output {output}
```

### 3. 输出格式

输出文件须为一个合法的 JSON 文件。建议采用下述结构表示计算图，字段命名建议直接沿用 ONNX 图中的原始名称（节点名与张量名）。

```json
{
  "format_version": "1.0",
  "graph_inputs":  [ { "name": "input",  "dtype": "FLOAT", "shape": ["batch", 1, 28, 28] } ],
  "graph_outputs": [ { "name": "logits", "dtype": "FLOAT", "shape": ["batch", 10] } ],
  "nodes": [
    {
      "name": "/fc1/Gemm",
      "op_type": "Gemm",
      "inputs": ["/flatten/Flatten_output_0", "fc1.weight", "fc1.bias"],
      "outputs": ["/fc1/Gemm_output_0"]
    }
  ],
  "edges": [
    { "src_node": "/flatten/Flatten", "dst_node": "/fc1/Gemm", "tensor": "/flatten/Flatten_output_0" }
  ]
}
```

推荐的字段含义如下：

| 字段 | 说明 |
|---|---|
| `graph_inputs` | 模型的输入张量列表（不含权重等 initializer） |
| `graph_outputs` | 模型的输出张量列表 |
| `nodes` | 计算图节点列表，每个节点包含节点名、算子类型、输入张量名列表、输出张量名列表 |
| `edges` | 数据依赖边列表，表示张量在节点间的流动 |

---

## C3.5

### 1. 任务描述

实现一个命令行程序，读取指定的 ONNX 模型与一批输入张量，在 GPU 上完成模型推理，并将推理结果写出。评测从精度、准确率、峰值显存与运行时间四个方面对结果进行考核。

### 2. 命令行接口

评测机以如下形式调用选手程序：

```
<选手程序> --onnx <model.onnx> --input <input_dir> --output <output_dir> [--batch-size N]
```

- `--onnx`：ONNX 模型文件路径。
- `--input`：输入目录，包含 `manifest.json` 与若干 `.npy` 文件（见第 4 节）。
- `--output`：输出目录，程序须将结果以相同的 `manifest.json` 加 `.npy` 格式写入（见第 5 节）。
- `--batch-size`：可选，评测机可能传入该参数，程序须能据此对样本分批推理。

报名时须提交命令模板，使用 `{onnx}`、`{input}`、`{output}` 作为占位符，例如：

```
python infer.py --onnx {onnx} --input {input} --output {output} --batch-size 256
```

### 3. 模型规格

评测使用三类模型，每类各一个公开版本（供调试）与一个隐藏版本（供评分）。两个版本结构相同、权重不同。

| 模型 | 任务 | 输入张量 | 输入形状 | 输出张量 | 输出形状 |
|---|---|---|---|---|---|
| MLP | MNIST 手写数字分类 | `input` (float32) | `[N, 1, 28, 28]` | `logits` (float32) | `[N, 10]` |
| ResNet-18（简化） | CIFAR-10 图像分类 | `input` (float32) | `[N, 3, 32, 32]` | `logits` (float32) | `[N, 10]` |
| Transformer（decoder-only） | 合成序列任务 | `input_ids` (int64) | `[N, 18]` | `logits` (float32) | `[N, 18, 14]` |

说明：

- 三个模型的批量维 `N` 均为动态维，支持任意批量大小。
- 输入数据已完成预处理：MNIST 与 CIFAR-10 图像已按标准均值与标准差归一化；Transformer 的 `input_ids` 为取值 0 至 13 的 token id。选手程序无须再做任何预处理，直接输入模型即可。
- 各模型使用的算子清单见第 8 节。

### 4. 输入格式

`<input_dir>/manifest.json` 描述输入目录中的各张量：

```json
{
  "tensors": [
    { "name": "input", "file": "input.npy", "dtype": "float32", "shape": [10000, 1, 28, 28] }
  ]
}
```

- 每个张量对应一个 `.npy` 文件，其第 0 维为样本数 `N`。
- `name` 为模型的输入张量名（MLP 与 ResNet 为 `input`，Transformer 为 `input_ids`），选手据此将张量对应到模型输入。


### 5. 输出格式

选手程序须在 `<output_dir>/` 下写入：

```
manifest.json
<output_name>.npy
```

- `manifest.json` 采用与输入相同的结构：`{"tensors": [{"name", "file", "dtype", "shape"}, ...]}`。
- 输出张量的 `name` 须使用模型的输出张量名（三个模型均为 `logits`）。
- 输出须覆盖全部 `N` 个样本，且第 0 维的顺序与输入一致（第 `i` 个输出对应第 `i` 个输入）。
- 输出 dtype 为 `float32`。


### 6. 评分规则

评测机对每个模型考核以下四项。其中精度与准确率为通过门槛，峰值显存与运行时间为排序与加分指标。

#### （1）精度测试（通过门槛）

将选手输出张量与标准答案逐元素比较。参考标准为 PyTorch 在 fp32 精度下计算的参考输出，随调试数据包一并提供（见第 9 节）。

通过条件为：

```
对所有元素： |out - golden| <= atol + rtol * |golden|
（等价于 numpy.allclose(out, golden, rtol, atol)）
```

阈值由各模型的 `thresholds.json` 给出，当前统一为 `rtol = atol = 1e-3`。

注意：标准答案以 fp32 精度计算。若在计算中使用 TF32、FP16、BF16 等低精度加速，ResNet 等较深网络的输出容易超出 1e-3 的阈值而导致精度不通过。如需以低精度换取性能，须自行确认精度仍在阈值范围内。

#### （2）准确率测试（分类模型的通过门槛）

MLP 与 ResNet：对输出 `logits` 取 argmax，与 `labels.npy` 中的真值标签比较，计算 top-1 准确率。

| 模型 | 准确率阈值 |
|---|---|
| MLP（MNIST） | ≥ 98% |
| ResNet-18（CIFAR-10） | ≥ 85% |

#### （3）峰值显存

评测机在选手程序运行期间，通过 NVML 按进程（含子进程）采样 GPU 已用显存并取峰值。

#### （4）运行时间

评测机记录选手程序从启动到退出的时间，作为性能指标。

#### 通过判定

精度测试通过，且准确率测试通过，即判定该模型的 C3.5 通过。峰值显存与运行时间不设硬性门槛，用于评分排序。

### 7. 支持的算子清单

下表列出三类模型导出后实际使用的全部算子（opset 17，共 17 种）。选手的推理程序须能执行这些算子。

| 模型 | 使用的算子 |
|---|---|
| MLP | `Flatten`、`Gemm`、`Relu` |
| ResNet-18（简化） | `Conv`、`Relu`、`Add`、`GlobalAveragePool`、`Flatten`、`Gemm` |
| Transformer | `Gather`、`Add`、`LayerNormalization`、`MatMul`、`Constant`、`Split`、`Reshape`、`Transpose`、`Div`、`Softmax`、`Erf`、`Mul` |

全部 17 种算子的并集为：`Add`、`Constant`、`Conv`、`Div`、`Erf`、`Flatten`、`Gather`、`Gemm`、`GlobalAveragePool`、`LayerNormalization`、`MatMul`、`Mul`、`Relu`、`Reshape`、`Softmax`、`Split`、`Transpose`。

补充说明：

- Transformer 的 GELU 被分解为 `Div`、`Erf`、`Add`、`Mul` 的组合，图中无单独的 Gelu 节点。
- `Gather` 仅用于 Transformer 的词嵌入查表（按 `input_ids` 索引词向量）。
- `Constant` 为图内嵌常量（如注意力缩放因子与因果掩码）。

### 8. 调试数据

每个公开模型均提供一份调试数据包，目录结构如下：

```
input/
  manifest.json      # 描述各输入张量（名称、文件、dtype、形状）
  <name>.npy         # 输入张量，第 0 维为样本数 N
golden/
  manifest.json
  <output>.npy       # 标准答案（PyTorch fp32 参考输出）
labels.npy           # 真值标签（仅分类模型：MLP 与 ResNet）
thresholds.json      # 该模型的精度与准确率阈值
```

选手可使用 `golden/` 自测精度、使用 `labels.npy` 自测准确率。评分所用的隐藏模型采用相同的目录结构。

### 9. 提交清单

提交材料须包含：

1. 程序源码及构建与运行说明。
2. 命令行模板：
   - C3.1：`... --onnx {onnx} --output {output}`
   - C3.5：`... --onnx {onnx} --input {input} --output {output} --batch-size 256`

建议在提交前，使用三个公开模型完成自测：对 C3.5，将输出与 `golden/` 用 `numpy.allclose(rtol=1e-3, atol=1e-3)` 比较精度，并用 `labels.npy` 核对准确率。
