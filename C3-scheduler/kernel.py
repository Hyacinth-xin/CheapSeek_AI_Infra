from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class KernelTuningParams:
    block_x: int
    block_y: int = 1
    block_z: int = 1
    grid_x: int = 1
    grid_y: int = 1
    grid_z: int = 1
    smem_bytes: int = -1
    shared_dtype: str = "fp32"


@dataclass
class KernelSpecRef:
    kernel_name: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    attrs: Dict = field(default_factory=dict)


@dataclass
class PrecisionProfile:
    precision: str
    supported_precisions: List[str] = field(default_factory=list)
    is_sensitive: bool = False
