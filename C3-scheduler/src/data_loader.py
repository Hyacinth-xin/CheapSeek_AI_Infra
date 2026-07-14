import json
import numpy as np
from pathlib import Path


def load_input(input_dir):
    input_dir = Path(input_dir)
    manifest_path = input_dir / "manifest.json"
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    tensors = {}
    for tensor_info in manifest["tensors"]:
        name = tensor_info["name"]
        file_path = input_dir / tensor_info["file"]
        dtype = tensor_info.get("dtype", "float32")
        
        arr = np.load(str(file_path))
        if dtype == "int64":
            arr = arr.astype(np.int64)
        elif dtype == "float32":
            arr = arr.astype(np.float32)
        
        tensors[name] = arr
    
    return tensors


def load_golden(golden_dir):
    golden_dir = Path(golden_dir)
    manifest_path = golden_dir / "manifest.json"
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    tensors = {}
    for tensor_info in manifest["tensors"]:
        name = tensor_info["name"]
        file_path = golden_dir / tensor_info["file"]
        arr = np.load(str(file_path)).astype(np.float32)
        tensors[name] = arr
    
    return tensors
