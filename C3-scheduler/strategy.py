from typing import List, Dict, Optional, Tuple
from graph import Graph, NodeInfo
from kernel import KernelSpecRef, KernelTuningParams, PrecisionProfile
from hardware import hardware


class DecompositionStrategy:
    def __init__(self):
        self.sensitive_ops = {"Softmax", "LayerNormalization", "BatchNormalization", 
                              "ReduceMax", "ReduceSum", "ReduceMean"}
        self._intermediate_counter = 0

    def _next_intermediate(self) -> str:
        self._intermediate_counter += 1
        return f"__c3_inter_{self._intermediate_counter}__"

    def select_precision(self, node: NodeInfo, graph: Graph) -> PrecisionProfile:
        if node.op_type in self.sensitive_ops:
            return PrecisionProfile(
                precision="fp32",
                supported_precisions=["fp32"],
                is_sensitive=True
            )
        
        all_supported = hardware.supported_precisions()
        if node.op_type in {"Gemm", "MatMul", "Conv"}:
            return PrecisionProfile(
                precision="fp16",
                supported_precisions=all_supported,
                is_sensitive=False
            )
        
        return PrecisionProfile(
            precision="fp32",
            supported_precisions=all_supported,
            is_sensitive=False
        )

    def decompose(self, node: NodeInfo, graph: Graph, precision: str) -> List[KernelSpecRef]:
        op_type = node.op_type
        kernels = []

        if op_type in {"Gemm", "MatMul"}:
            kernel_name = f"matmul_{precision}"
            kernels.append(KernelSpecRef(
                kernel_name=kernel_name,
                inputs=node.inputs,
                outputs=node.outputs
            ))

        elif op_type == "Softmax":
            inter1 = self._next_intermediate()
            inter2 = self._next_intermediate()
            inter3 = self._next_intermediate()
            
            kernels.append(KernelSpecRef(
                kernel_name="reduce_max",
                inputs=node.inputs,
                outputs=[inter1]
            ))
            kernels.append(KernelSpecRef(
                kernel_name="exp",
                inputs=[inter1],
                outputs=[inter2]
            ))
            kernels.append(KernelSpecRef(
                kernel_name="reduce_sum",
                inputs=[inter2],
                outputs=[inter3]
            ))
            kernels.append(KernelSpecRef(
                kernel_name="div",
                inputs=[inter2, inter3],
                outputs=node.outputs
            ))

        elif op_type == "LayerNormalization":
            inter1 = self._next_intermediate()
            inter2 = self._next_intermediate()
            inter3 = self._next_intermediate()
            
            kernels.append(KernelSpecRef(
                kernel_name="reduce_mean",
                inputs=node.inputs,
                outputs=[inter1]
            ))
            kernels.append(KernelSpecRef(
                kernel_name="sub",
                inputs=node.inputs + [inter1],
                outputs=[inter2]
            ))
            kernels.append(KernelSpecRef(
                kernel_name="mul",
                inputs=[inter2],
                outputs=[inter3]
            ))
            kernels.append(KernelSpecRef(
                kernel_name="sqrt",
                inputs=[inter3],
                outputs=node.outputs
            ))

        elif op_type == "Conv":
            kernel_name = f"im2col_{precision}"
            kernels.append(KernelSpecRef(
                kernel_name=kernel_name,
                inputs=node.inputs,
                outputs=node.outputs
            ))

        elif op_type in {"Relu", "Add", "Mul", "Div", "Erf"}:
            kernel_name = f"{op_type.lower()}_{precision}"
            kernels.append(KernelSpecRef(
                kernel_name=kernel_name,
                inputs=node.inputs,
                outputs=node.outputs
            ))

        elif op_type in {"Flatten", "Reshape", "Transpose", "Split"}:
            kernel_name = f"{op_type.lower()}_fp32"
            kernels.append(KernelSpecRef(
                kernel_name=kernel_name,
                inputs=node.inputs,
                outputs=node.outputs
            ))

        elif op_type == "Gather":
            kernels.append(KernelSpecRef(
                kernel_name="gather_fp32",
                inputs=node.inputs,
                outputs=node.outputs
            ))

        elif op_type == "Constant":
            kernels.append(KernelSpecRef(
                kernel_name="constant_fp32",
                inputs=[],
                outputs=node.outputs
            ))

        elif op_type == "GlobalAveragePool":
            kernels.append(KernelSpecRef(
                kernel_name="reduce_mean",
                inputs=node.inputs,
                outputs=node.outputs
            ))

        else:
            kernels.append(KernelSpecRef(
                kernel_name=f"generic_{precision}",
                inputs=node.inputs,
                outputs=node.outputs
            ))

        return kernels

    def tune_kernel(self, ref: KernelSpecRef, precision: str, problem_size: Tuple[int, ...]) -> KernelTuningParams:
        kernel_name = ref.kernel_name
        
        if kernel_name.startswith("matmul"):
            if len(problem_size) >= 3:
                M, N, K = problem_size[0], problem_size[1], problem_size[2]
            else:
                M, N, K = 256, 256, 256
            
            block_x = min(256, hardware.max_threads_per_block)
            grid_x = (M * N + block_x - 1) // block_x
            smem_bytes = 2 * block_x * 16
            
            return KernelTuningParams(
                block_x=block_x,
                grid_x=grid_x,
                smem_bytes=smem_bytes
            )

        elif kernel_name.startswith("conv") or kernel_name.startswith("im2col"):
            block_x = min(256, hardware.max_threads_per_block)
            grid_x = max(1, (problem_size[0] * problem_size[1]) // block_x)
            
            return KernelTuningParams(
                block_x=block_x,
                grid_x=grid_x,
                smem_bytes=hardware.smem_bytes
            )

        elif kernel_name.startswith("reduce"):
            block_x = min(1024, hardware.max_threads_per_block)
            grid_x = max(1, problem_size[0] // block_x)
            
            return KernelTuningParams(
                block_x=block_x,
                grid_x=grid_x,
                smem_bytes=0
            )

        elif kernel_name in {"exp", "sub", "mul", "div", "sqrt", "erf", "relu"}:
            block_x = min(256, hardware.max_threads_per_block)
            total_elements = 1
            for dim in problem_size:
                if isinstance(dim, int) and dim > 0:
                    total_elements *= dim
            grid_x = max(1, (total_elements + block_x - 1) // block_x)
            
            return KernelTuningParams(
                block_x=block_x,
                grid_x=grid_x,
                smem_bytes=0
            )

        else:
            block_x = min(256, hardware.max_threads_per_block)
            grid_x = 1
            
            return KernelTuningParams(
                block_x=block_x,
                grid_x=grid_x,
                smem_bytes=0
            )


strategy = DecompositionStrategy()
