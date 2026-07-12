# 数据文件准备说明

本仓库使用 Git 管理代码源码。部分官方提供的大型测试数据文件（二进制格式）体积较大，未纳入版本控制。
克隆仓库后，请按照本文档准备好所需的数据文件。

---

## 总览

| 模块 | 是否需要额外数据 | 说明 |
|------|-----------------|------|
| C1-compiler | 不需要 | 测试用例（.ptx）已在仓库中 |
| C2-runtime | 不需要 | 设备库与 kernel 镜像已在仓库中 |
| C3-scheduler | **需要** | ONNX 模型与测试数据较大，需单独准备 |

---

## C3-scheduler 数据准备

### 1. 模型文件（ONNX）

**目录**：`C3-scheduler/models/`

需要放置以下 3 个 ONNX 模型文件：

```
C3-scheduler/models/
├── mlp_v1.onnx            # MLP 模型（MNIST 手写数字分类）
├── resnet_v1.onnx         # 简化版 ResNet-18（CIFAR-10 图像分类）
└── transformer_v1.onnx    # decoder-only Transformer（合成序列任务）
```

**用途**：
- C3.1（计算图解析）的输入
- C3.5（端到端推理）的模型输入

---

### 2. 测试数据（.npy）

**目录**：`C3-scheduler/testdata/c35/`

三个模型各有一套测试数据，结构如下：

```
C3-scheduler/testdata/c35/
├── mlp_v1/
│   ├── input/
│   │   ├── manifest.json      # （已在仓库中）
│   │   └── input.npy          # 输入张量
│   ├── golden/
│   │   ├── manifest.json      # （已在仓库中）
│   │   └── logits.npy         # 标准答案（PyTorch fp32 参考输出）
│   ├── labels.npy             # 真值标签
│   └── thresholds.json        # （已在仓库中）
├── resnet_v1/
│   ├── input/
│   │   ├── manifest.json      # （已在仓库中）
│   │   └── input.npy          # 输入张量
│   ├── golden/
│   │   ├── manifest.json      # （已在仓库中）
│   │   └── logits.npy         # 标准答案
│   ├── labels.npy             # 真值标签
│   └── thresholds.json        # （已在仓库中）
└── transformer_v1/
    ├── input/
    │   ├── manifest.json      # （已在仓库中）
    │   └── input_ids.npy      # 输入张量
    ├── golden/
    │   ├── manifest.json      # （已在仓库中）
    │   └── logits.npy         # 标准答案
    ├── labels.npy             # 真值标签
    └── thresholds.json        # （已在仓库中）
```

> **注意**：各目录下的 `manifest.json` 和 `thresholds.json` 是文本描述文件，已在仓库中。
> 需要补充的是 `.npy` 二进制数据文件。

**用途**：
- 自测 C3.5 端到端推理的正确性与精度
- 对比输出与 golden 标准答案：`np.allclose(out, golden, rtol=1e-3, atol=1e-3)`
- 用 `labels.npy` 计算分类准确率（MLP ≥ 98%，ResNet ≥ 85%）

---

## 获取方式

上述模型文件与测试数据均为竞赛官方提供的公开资料包内容，请从以下渠道获取：

1. **竞赛官方发布的 C3 选手资料包**（starter-kit / release package）
2. 组队群内共享的资料文件

下载后，按照上面的目录结构，将文件放入对应路径即可。

---

## 验证数据是否齐全

准备完成后，目录结构应该满足：

```bash
# 检查模型文件
ls C3-scheduler/models/*.onnx
# 应输出 3 个文件

# 检查 npy 数据文件
find C3-scheduler/testdata -name "*.npy" | wc -l
# 应输出 11 个文件（3 个模型 × (input + golden + labels) = 9，
# 注意 transformer 的 input 文件名是 input_ids.npy）
```
