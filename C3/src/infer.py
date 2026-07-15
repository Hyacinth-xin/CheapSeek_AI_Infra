#!/usr/bin/env python3
"""C3.5 端到端推理 CLI 入口。

用法（符合 COMPETITOR_GUIDE.md 提交模板）：
    python infer.py --onnx <model.onnx> --input <input_dir> --output <output_dir> --batch-size 256

提交模板：
    python infer.py --onnx {onnx} --input {input} --output {output} --batch-size 256
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.executor import InferenceExecutor


def main():
    parser = argparse.ArgumentParser(description="C3.5 ONNX model inference")
    parser.add_argument("--onnx", required=True, help="ONNX model path")
    parser.add_argument("--input", required=True, help="Input directory with manifest.json + .npy")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for batched inference (default: 256)")
    args = parser.parse_args()

    executor = InferenceExecutor(args.onnx, batch_size=args.batch_size)

    try:
        samples = executor.batch_run(args.input, args.output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Inference completed: {samples} samples")
    sys.exit(0)


if __name__ == "__main__":
    main()
