#!/usr/bin/env python3
"""C3.5 服务器 cupy 验证脚本：精度 + 准确率 + 计时。

用法（在 C3-scheduler 目录下）：
    python tests/test_c35_cupy.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.backend import is_gpu, backend_name
from src.executor import InferenceExecutor
from src.data_loader import load_golden

print(f"=== Backend: {backend_name()} | is_gpu={is_gpu()} ===\n")

MODELS = [
    ("mlp_v1", 0.98),
    ("transformer_v1", None),
    ("resnet_v1", 0.85),
]

all_pass = True
for name, thr in MODELS:
    t0 = time.time()
    ex = InferenceExecutor(f"models/{name}.onnx")
    ex._parse_model()
    out = ex.predict(f"testdata/c35/{name}/input", batch_size=256)
    elapsed = time.time() - t0

    logits = out["logits"]
    golden = load_golden(f"testdata/c35/{name}/golden")["logits"]
    max_diff = float(np.max(np.abs(logits - golden)))
    precision_ok = np.allclose(logits, golden, rtol=1e-3, atol=1e-3)

    line = f"{name}: precision={'PASS' if precision_ok else 'FAIL'}"
    line += f" | max_diff={max_diff:.2e} | {elapsed:.2f}s"

    if thr is not None:
        labels = np.argmax(logits, axis=1)
        true = np.load(f"testdata/c35/{name}/labels.npy")
        acc = float((labels == true).mean())
        acc_ok = acc >= thr
        line += f" | acc={acc:.4f} (>={thr}) {'PASS' if acc_ok else 'FAIL'}"
        all_pass = all_pass and acc_ok

    print(line)
    all_pass = all_pass and precision_ok

print(f"\n=== {'ALL PASS' if all_pass else 'SOME FAILED'} ===")
sys.exit(0 if all_pass else 1)
