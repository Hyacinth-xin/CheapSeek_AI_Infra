import json
import numpy as np
from pathlib import Path


def write_output(output_dir, outputs):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_tensors = []
    for name, arr in outputs.items():
        file_name = f"{name}.npy"
        np.save(str(output_dir / file_name), arr)
        
        manifest_tensors.append({
            "name": name,
            "file": file_name,
            "dtype": "float32",
            "shape": list(arr.shape)
        })
    
    manifest = {"tensors": manifest_tensors}
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    return sum(arr.shape[0] for arr in outputs.values())
