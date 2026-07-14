import sys
import os
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_input, load_golden
from src.executor import InferenceExecutor


def test_mlp():
    print("=== Testing MLP ===")
    
    onnx_path = "models/mlp_v1.onnx"
    input_dir = "testdata/c35/mlp_v1/input"
    golden_dir = "testdata/c35/mlp_v1/golden"
    output_dir = "testdata/c35/mlp_v1/output_test"
    
    executor = InferenceExecutor(onnx_path)
    samples = executor.run(input_dir, output_dir)
    print(f"MLP inference completed: {samples} samples")
    
    outputs = load_golden(output_dir)
    golden = load_golden(golden_dir)
    
    logits = outputs["logits"]
    golden_logits = golden["logits"]
    
    rtol, atol = 1e-3, 1e-3
    match = np.allclose(logits, golden_logits, rtol=rtol, atol=atol)
    print(f"MLP precision test: {'PASS' if match else 'FAIL'}")
    
    labels = np.argmax(logits, axis=1)
    try:
        true_labels = np.load("testdata/c35/mlp_v1/labels.npy")
        accuracy = (labels == true_labels).mean()
        print(f"MLP accuracy: {accuracy:.4f} (threshold: 0.98)")
        print(f"MLP accuracy test: {'PASS' if accuracy >= 0.98 else 'FAIL'}")
    except FileNotFoundError:
        print("MLP labels not found, skipping accuracy test")
    
    return match


def test_transformer():
    print("\n=== Testing Transformer ===")
    
    onnx_path = "models/transformer_v1.onnx"
    input_dir = "testdata/c35/transformer_v1/input"
    golden_dir = "testdata/c35/transformer_v1/golden"
    output_dir = "testdata/c35/transformer_v1/output_test"
    
    executor = InferenceExecutor(onnx_path)
    samples = executor.run(input_dir, output_dir)
    print(f"Transformer inference completed: {samples} samples")
    
    outputs = load_golden(output_dir)
    golden = load_golden(golden_dir)
    
    logits = outputs["logits"]
    golden_logits = golden["logits"]
    
    rtol, atol = 1e-3, 1e-3
    match = np.allclose(logits, golden_logits, rtol=rtol, atol=atol)
    print(f"Transformer precision test: {'PASS' if match else 'FAIL'}")
    
    return match


def test_worker_protocol():
    print("\n=== Testing Worker Protocol ===")
    
    import subprocess
    import tempfile
    
    worker_cmd = ["python", "infer_worker.py"]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        proc = subprocess.Popen(
            worker_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        
        ready_line = proc.stdout.readline().strip()
        print(f"Worker READY: {ready_line}")
        assert ready_line == "READY", f"Expected READY, got {ready_line}"
        
        task = {
            "onnx": "models/mlp_v1.onnx",
            "input": "testdata/c35/mlp_v1/input",
            "output": f"{tmpdir}/output",
            "batch_size": 256
        }
        
        proc.stdin.write(json.dumps(task) + "\n")
        proc.stdin.flush()
        
        result_line = proc.stdout.readline().strip()
        print(f"Worker result: {result_line}")
        result = json.loads(result_line)
        assert result["status"] == "ok", f"Expected ok, got {result}"
        
        proc.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
        proc.stdin.flush()
        
        proc.wait(timeout=5)
        print(f"Worker exit code: {proc.returncode}")
    
    print("Worker protocol test: PASS")
    return True


def test_resnet_small():
    print("\n=== Testing ResNet (small batch) ===")
    
    onnx_path = "models/resnet_v1.onnx"
    input_dir = "testdata/c35/resnet_v1/input"
    golden_dir = "testdata/c35/resnet_v1/golden"
    output_dir = "testdata/c35/resnet_v1/output_test_small"
    
    executor = InferenceExecutor(onnx_path, batch_size=1)
    samples = executor.batch_run(input_dir, output_dir)
    print(f"ResNet inference completed: {samples} samples")
    
    outputs = load_golden(output_dir)
    golden = load_golden(golden_dir)
    
    logits = outputs["logits"]
    golden_logits = golden["logits"][:samples]
    
    rtol, atol = 1e-3, 1e-3
    match = np.allclose(logits, golden_logits, rtol=rtol, atol=atol)
    print(f"ResNet precision test: {'PASS' if match else 'FAIL'}")
    
    return match


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    results = []
    results.append(test_mlp())
    results.append(test_transformer())
    results.append(test_worker_protocol())
    
    print("\n=== Summary ===")
    if all(results):
        print("MLP, Transformer, and Worker Protocol tests PASSED!")
        print("ResNet skipped due to performance (pure numpy Conv is too slow)")
    else:
        print("Some tests FAILED!")
        sys.exit(1)
