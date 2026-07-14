#!/usr/bin/env python3
"""C3.5 端到端测试：Worker 协议验证（三模型）+ 直接推理检查。

Worker 协议是评测通道，必须覆盖全部三类模型。
本地用 numpy 回退运行，服务器用 cupy+GPU。
"""
import sys
import os
import json
import subprocess
import tempfile
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_input, load_golden
from src.executor import InferenceExecutor


MODELS = [
    ("mlp_v1", 0.98),
    ("transformer_v1", None),
    ("resnet_v1", 0.85),
]


def test_direct_predict():
    """直接 predict() 遍历三模型，验证精度+准确率（无 worker）。"""
    print("=== Direct predict (3 models) ===")
    all_ok = True
    for name, thr in MODELS:
        ex = InferenceExecutor(f"models/{name}.onnx")
        ex._parse_model()
        out = ex.predict(f"testdata/c35/{name}/input", batch_size=256)
        logits = out["logits"]

        golden = load_golden(f"testdata/c35/{name}/golden")["logits"]
        match = np.allclose(logits, golden, rtol=1e-3, atol=1e-3)

        line = f"  {name}: precision={'PASS' if match else 'FAIL'}"
        if thr is not None:
            labels = np.argmax(logits, axis=1)
            true = np.load(f"testdata/c35/{name}/labels.npy")
            acc = float((labels == true).mean())
            line += f" | acc={acc:.4f} (>={thr}) {'PASS' if acc >= thr else 'FAIL'}"
            all_ok = all_ok and match and acc >= thr
        else:
            all_ok = all_ok and match
        print(line)

    return all_ok


def test_worker_protocol():
    """启动一次 worker，对三模型各发一个任务，验证协议+精度+准确率。

    验证要点（对应 C35_WORKER_PROTOCOL.md）：
    - READY 握手
    - 多模型复用同一 worker
    - batch_size=256 分批
    - exit 后干净退出
    """
    print("\n=== Worker Protocol (3 models) ===")
    worker_cmd = ["python", "infer_worker.py"]

    with tempfile.TemporaryDirectory() as tmpdir:
        proc = subprocess.Popen(
            worker_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )

        # Step 1: READY 握手
        ready_line = proc.stdout.readline().strip()
        assert ready_line == "READY", f"Expected READY, got {ready_line}"
        print("  READY: OK")

        # Step 2: 依次发送三模型任务
        all_ok = True
        for name, thr in MODELS:
            output_dir = f"{tmpdir}/{name}"
            task = {
                "onnx": f"models/{name}.onnx",
                "input": f"testdata/c35/{name}/input",
                "output": output_dir,
                "batch_size": 256,
            }
            proc.stdin.write(json.dumps(task) + "\n")
            proc.stdin.flush()

            result_line = proc.stdout.readline().strip()
            result = json.loads(result_line)
            assert result["status"] == "ok", f"  {name}: worker returned error: {result}"

            # 验证输出精度
            outputs = load_golden(output_dir)
            golden = load_golden(f"testdata/c35/{name}/golden")
            logits = outputs["logits"]
            golden_logits = golden["logits"]
            match = np.allclose(logits, golden_logits, rtol=1e-3, atol=1e-3)

            line = f"  {name}: precision={'PASS' if match else 'FAIL'}"
            if thr is not None:
                labels = np.argmax(logits, axis=1)
                true = np.load(f"testdata/c35/{name}/labels.npy")
                acc = float((labels == true).mean())
                line += f" | acc={acc:.4f} (>={thr}) {'PASS' if acc >= thr else 'FAIL'}"
                all_ok = all_ok and match and acc >= thr
            else:
                all_ok = all_ok and match

            print(line)

        # Step 3: 退出
        proc.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
        proc.stdin.flush()
        proc.wait(timeout=10)
        print(f"  Worker exit code: {proc.returncode}")

    print(f"  Worker protocol: {'ALL PASS' if all_ok else 'FAILED'}")
    return all_ok


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    results = []
    results.append(test_direct_predict())
    results.append(test_worker_protocol())

    print("\n=== Summary ===")
    if all(results):
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
        sys.exit(1)
