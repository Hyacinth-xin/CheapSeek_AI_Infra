# C3 赛题 · 选手资料包（C3.1 / C3.5）

本资料包供参赛选手开发与自测使用。

## 目录结构

```
release_to_competitors/
├── docs/
│   └── COMPETITOR_GUIDE.md        # 赛题与评测规范（必读）
├── models/                        # 3 个公开模型（用于本地开发与调试）
│   ├── mlp_v1.onnx                # MLP，MNIST 手写数字分类
│   ├── resnet_v1.onnx             # 简化版 ResNet-18，CIFAR-10 图像分类
│   └── transformer_v1.onnx        # decoder-only Transformer，合成序列任务
└── testdata/
    └── c35/                       # C3.5 各公开模型的调试数据包
        ├── mlp_v1/
        │   ├── input/             # 输入张量（manifest.json + .npy）
        │   ├── golden/            # 标准答案（PyTorch fp32 参考输出）
        │   ├── labels.npy         # 真值标签（分类模型）
        │   └── thresholds.json    # 精度与准确率阈值
        ├── resnet_v1/
        └── transformer_v1/
```

## 快速自测（C3.5）

以 ResNet 为例，用你的推理程序产出输出后，与标准答案比对：

```python
import numpy as np
out  = np.load("your_output/logits.npy")
gold = np.load("testdata/c35/resnet_v1/golden/logits.npy")
lab  = np.load("testdata/c35/resnet_v1/labels.npy")
print("precision pass:", np.allclose(out, gold, rtol=1e-3, atol=1e-3))
print("accuracy:", (out.argmax(1) == lab).mean())
```
