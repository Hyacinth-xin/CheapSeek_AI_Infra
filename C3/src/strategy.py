from typing import List, Dict, Optional, Tuple
from .graph import Graph, NodeInfo
from .kernel import KernelSpecRef, KernelTuningParams, PrecisionProfile
from .hardware import hardware


class DecompositionStrategy:
    """算子分解与内核选择策略"""
    def __init__(self):
        self.sensitive_ops = {"Softmax", "LayerNormalization", "BatchNormalization",
                              "ReduceMax", "ReduceSum", "ReduceMean"}
        self._intermediate_counter = 0
        # 精度策略：根据算子类型映射到不同精度
        self._precision_strategy = {
            # GEMM/MatMul 可以用 fp8/fp4
            "Gemm": ["fp8", "fp4", "fp16"],
            "MatMul": ["fp8", "fp4", "fp16"],
            # Conv 用 fp16 更保守
            "Conv": ["fp16", "fp8"],
            # Elementwise 用 fp16
            "Add": ["fp16"],
            "Mul": ["fp16"],
            "Div": ["fp16"],
            "Sub": ["fp16"],
            "Relu": ["fp16"],
            "Erf": ["fp32"],  # Erf 对精度敏感
        }
        self._precision_counter = {"fp32": 0, "fp16": 0, "fp8": 0, "fp4": 0}
        self._round_counter = 0  # 独立轮询计数器，确保精度覆盖多样化

    def _next_intermediate(self) -> str:
        self._intermediate_counter += 1
        return f"__c3_inter_{self._intermediate_counter}__"

    def _get_problem_size(self, node: NodeInfo, graph: Graph) -> Tuple[int, ...]:
        """从算子节点和图中推断问题规模"""
        if node.outputs:
            output_tensor = graph.get_tensor(node.outputs[0])
            if output_tensor and output_tensor.shape:
                dims = [d for d in output_tensor.shape if isinstance(d, int) and d > 0]
                if dims:
                    return tuple(dims)

        if node.op_type in {"Gemm", "MatMul"}:
            return (256, 256, 256)
        elif node.op_type == "Conv":
            return (32, 32)
        else:
            return (1024,)

    def decompose_with_tuning(self, node: NodeInfo, graph: Graph, precision: str) -> List[KernelSpecRef]:
        """分解算子并附带调优参数（便捷方法）"""
        kernels = self.decompose(node, graph, precision)
        problem_size = self._get_problem_size(node, graph)

        for kernel_ref in kernels:
            tuning_params = self.tune_kernel(kernel_ref, precision, problem_size)
            kernel_ref.tuning_params = tuning_params

        return kernels

    def select_precision(self, node: NodeInfo, graph: Graph) -> PrecisionProfile:
        if node.op_type in self.sensitive_ops:
            self._precision_counter["fp32"] += 1
            return PrecisionProfile(
                precision="fp32",
                supported_precisions=["fp32"],
                is_sensitive=True
            )

        all_supported = hardware.supported_precisions()

        # 根据策略选择精度
        if node.op_type in self._precision_strategy:
            precisions = self._precision_strategy[node.op_type]
            # 用独立计数器轮询，确保每种精度都被覆盖
            idx = self._round_counter % len(precisions)
            selected = precisions[idx]
            self._round_counter += 1
            self._precision_counter[selected] += 1
            return PrecisionProfile(
                precision=selected,
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
            # 根据卷积核大小选择策略：3x3 用 Winograd，其他用 im2col
            kernel_size = node.attrs.get("kernel_shape", [3, 3])
            use_winograd = len(kernel_size) >= 2 and kernel_size[0] == 3 and kernel_size[1] == 3

            if use_winograd:
                kernel_name = f"winograd_forward_3x3_{precision}"
            else:
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
