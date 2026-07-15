"""ONNX 推理执行器（numpy/cupy 双后端）。

- numpy 后端：本地初步测试逻辑正确性
- cupy 后端：服务器 GPU 推理，满足 NVML 显存采样与性能评测

权重在 _parse_model 时预上传到设备（initializers_gpu），体现 C3.4 权重预加载路径。
支持 C3.5 全部 17 种算子。
"""

import numpy as np
import onnx
from pathlib import Path

from . import backend
from .backend import xp, as_device, as_host, is_gpu


class ONNXExecutor:
    def __init__(self, onnx_path, batch_size=None):
        self.onnx_path = Path(onnx_path)
        self.batch_size = batch_size
        self.model = None
        self.graph = None
        self.input_names = None
        self.output_names = None
        self.value_info = {}
        # host 侧原始权重
        self.initializers = {}
        # 设备侧权重缓存（预上传），体现 C3.4 权重预加载
        self.initializers_gpu = {}

    def _parse_model(self):
        self.model = onnx.load(str(self.onnx_path))
        self.graph = self.model.graph

        self.input_names = [inp.name for inp in self.graph.input]
        self.output_names = [out.name for out in self.graph.output]

        for vi in self.graph.value_info:
            shape = []
            for d in vi.type.tensor_type.shape.dim:
                if d.dim_value:
                    shape.append(d.dim_value)
                else:
                    shape.append(None)
            self.value_info[vi.name] = shape

        # 权重预加载：解析时即上传到设备
        for init in self.graph.initializer:
            arr = onnx.numpy_helper.to_array(init).astype(np.float32)
            self.initializers[init.name] = arr
            self.initializers_gpu[init.name] = as_device(arr)

    # ------------------------------------------------------------------
    # 算子实现（全部基于 xp，numpy/cupy 通用）
    # ------------------------------------------------------------------
    def _execute_node(self, node, tensors):
        op_type = node.op_type

        inputs = []
        for inp_name in node.input:
            if inp_name in tensors:
                inputs.append(tensors[inp_name])
            elif inp_name in self.initializers_gpu:
                inputs.append(self.initializers_gpu[inp_name])
            else:
                inputs.append(None)

        if op_type == "Add":
            result = inputs[0] + inputs[1]
        elif op_type == "Mul":
            result = inputs[0] * inputs[1]
        elif op_type == "Div":
            result = inputs[0] / inputs[1]
        elif op_type == "Sub":
            result = inputs[0] - inputs[1]
        elif op_type == "Relu":
            result = xp.maximum(inputs[0], 0)
        elif op_type == "Erf":
            result = backend.erf(inputs[0])
        elif op_type == "MatMul":
            result = inputs[0] @ inputs[1]
        elif op_type == "Gemm":
            a, b = inputs[0], inputs[1]
            c = inputs[2] if len(inputs) > 2 else None
            trans_a, trans_b = False, True
            for attr in node.attribute:
                if attr.name == "transA":
                    trans_a = attr.i == 1
                if attr.name == "transB":
                    trans_b = attr.i == 1
            if trans_a:
                a = a.T
            if trans_b:
                b = b.T
            result = a @ b
            if c is not None:
                result = result + c
        elif op_type == "Softmax":
            axis = 1
            for attr in node.attribute:
                if attr.name == "axis":
                    axis = attr.i
            x = inputs[0]
            exp_x = xp.exp(x - xp.max(x, axis=axis, keepdims=True))
            result = exp_x / xp.sum(exp_x, axis=axis, keepdims=True)
        elif op_type == "LayerNormalization":
            x, scale, bias = inputs[0], inputs[1], inputs[2]
            eps = 1e-5
            for attr in node.attribute:
                if attr.name == "epsilon":
                    eps = attr.f
            mean = xp.mean(x, axis=-1, keepdims=True)
            var = xp.var(x, axis=-1, keepdims=True)
            norm = (x - mean) / xp.sqrt(var + eps)
            result = norm * scale + bias
        elif op_type == "Reshape":
            x, shape_tensor = inputs[0], inputs[1]
            if shape_tensor is None:
                shape = []
                for attr in node.attribute:
                    if attr.name == "shape":
                        shape = list(attr.ints)
            else:
                shape = as_host(shape_tensor).astype(np.int64).tolist()

            input_shape = x.shape
            # ONNX 语义：0 = 沿用输入对应维度；-1 = 推断
            new_shape = []
            for idx, dim in enumerate(shape):
                if dim == 0:
                    new_shape.append(input_shape[idx])
                else:
                    new_shape.append(dim)

            if -1 in new_shape:
                total = 1
                for d in input_shape:
                    total *= d
                known = 1
                for d in new_shape:
                    if d != -1:
                        known *= d
                new_shape = [d if d != -1 else total // known for d in new_shape]

            result = x.reshape(new_shape)
        elif op_type == "Transpose":
            x = inputs[0]
            perm = list(range(x.ndim))
            for attr in node.attribute:
                if attr.name == "perm":
                    perm = list(attr.ints)
            result = xp.transpose(x, perm)
        elif op_type == "Flatten":
            x = inputs[0]
            axis = 1
            for attr in node.attribute:
                if attr.name == "axis":
                    axis = attr.i
            if axis == 0:
                result = x.reshape(1, -1)
            else:
                result = x.reshape(x.shape[0], -1)
        elif op_type == "Split":
            x = inputs[0]
            split = []
            axis = 0
            for attr in node.attribute:
                if attr.name == "split":
                    split = list(attr.ints)
                if attr.name == "axis":
                    axis = attr.i

            if len(split) == 0:
                num_outputs = len(node.output)
                total = x.shape[axis]
                each = total // num_outputs
                split = [each] * (num_outputs - 1) + [total - each * (num_outputs - 1)]

            indices = xp.cumsum(xp.array(split[:-1])).tolist()
            result = xp.split(x, indices, axis=axis)
        elif op_type == "Gather":
            x, indices = inputs[0], inputs[1]
            axis = 0
            for attr in node.attribute:
                if attr.name == "axis":
                    axis = attr.i
            idx = indices.astype(xp.int64)
            result = xp.take(x, idx, axis=axis)
        elif op_type == "Constant":
            value = None
            for attr in node.attribute:
                if attr.name == "value":
                    arr = onnx.numpy_helper.to_array(attr.t)
                    value = as_device(arr.astype(np.float32))
            result = value
        elif op_type == "Conv":
            result = self._conv(inputs, node)
        elif op_type == "GlobalAveragePool":
            x = inputs[0]
            result = xp.mean(x, axis=(2, 3))
        elif op_type == "Identity":
            # BigFormer: 直通复制，output = input
            result = inputs[0]
        else:
            raise NotImplementedError(f"Unsupported operator: {op_type}")

        return result

    # ------------------------------------------------------------------
    # Conv: im2col + matmul（numpy/cupy 通用）
    #
    # 从 padded 输入直接展开为 3D col 矩阵，避免 5D 中间张量。
    # 内存峰值 = N * C_in * KH * KW * out_h * out_w（与 3D col 相同）。
    # ------------------------------------------------------------------
    def _conv(self, inputs, node):
        x, w = inputs[0], inputs[1]
        b = inputs[2] if len(inputs) > 2 else None
        stride = [1, 1]
        padding = [0, 0, 0, 0]
        dilation = [1, 1]
        groups = 1

        for attr in node.attribute:
            if attr.name == "strides":
                stride = list(attr.ints)
            if attr.name == "pads":
                padding = list(attr.ints)
            if attr.name == "dilations":
                dilation = list(attr.ints)
            if attr.name == "group":
                groups = attr.i

        if len(padding) == 2:
            padding = [padding[0], padding[1], padding[0], padding[1]]
        if len(stride) == 1:
            stride = [stride[0], stride[0]]

        N, C_in, H_in, W_in = x.shape
        C_out, C_in_g, KH, KW = w.shape
        stride_h, stride_w = stride
        dilation_h, dilation_w = dilation

        pad_top, pad_left = padding[0], padding[1]
        pad_bottom, pad_right = padding[2], padding[3]

        H_pad = H_in + pad_top + pad_bottom
        W_pad = W_in + pad_left + pad_right
        out_h = (H_pad - dilation_h * (KH - 1) - 1) // stride_h + 1
        out_w = (W_pad - dilation_w * (KW - 1) - 1) // stride_w + 1

        # zero padding
        if any(padding):
            x_padded = xp.zeros((N, C_in, H_pad, W_pad), dtype=x.dtype)
            x_padded[:, :, pad_top:pad_top + H_in, pad_left:pad_left + W_in] = x
        else:
            x_padded = x

        # im2col 到 5D → reshape 3D（5D 和 3D 共享内存，零开销）
        # KH*KW 双循环（9 次大块拷贝，比 576 次小块快 5-10x）
        col = xp.empty((N, C_in, KH, KW, out_h, out_w), dtype=x.dtype)
        for i in range(KH):
            i_s = i * dilation_h
            for j in range(KW):
                j_s = j * dilation_w
                col[:, :, i, j, :, :] = x_padded[:, :,
                                                  i_s:i_s + stride_h * out_h:stride_h,
                                                  j_s:j_s + stride_w * out_w:stride_w]

        if groups == 1:
            col = col.reshape(N, C_in * KH * KW, out_h * out_w)
            w_mat = w.reshape(C_out, C_in * KH * KW)
            out = xp.matmul(w_mat, col).reshape(N, C_out, out_h, out_w)
        else:
            col = col.reshape(N, C_in * KH * KW, out_h * out_w)
            out = xp.empty((N, C_out, out_h, out_w), dtype=x.dtype)
            C_out_g = C_out // groups
            C_in_per_g = C_in // groups
            K_g = C_in_per_g * KH * KW
            for g in range(groups):
                col_g = col[:, g * K_g:(g + 1) * K_g, :]
                w_g = w[g * C_out_g:(g + 1) * C_out_g].reshape(C_out_g, K_g)
                out[:, g * C_out_g:(g + 1) * C_out_g, :, :] = \
                    xp.matmul(w_g, col_g).reshape(N, C_out_g, out_h, out_w)

        if b is not None:
            out = out + b.reshape(1, -1, 1, 1)
        return out

    # ------------------------------------------------------------------
    # 图执行
    # ------------------------------------------------------------------
    def _run_graph(self, input_tensors):
        tensors = {}
        # 输入上设备；int64 (input_ids) 保留 int64，其余 float32
        for name, arr in input_tensors.items():
            host = as_host(arr)
            if host.dtype == np.int64:
                tensors[name] = as_device(host.astype(np.int64))
            else:
                tensors[name] = as_device(host.astype(np.float32))

        for node in self.graph.node:
            result = self._execute_node(node, tensors)

            if isinstance(result, (tuple, list)):
                for i, out_name in enumerate(node.output):
                    if out_name:
                        tensors[out_name] = result[i]
            else:
                for out_name in node.output:
                    if out_name:
                        tensors[out_name] = result

        outputs = {}
        for name in self.output_names:
            if name in tensors:
                outputs[name] = tensors[name].astype(xp.float32)
        return outputs

    def predict(self, input_dir, batch_size=None):
        """分批推理，返回拼接好的 host numpy dict（不写盘）。

        batch_size=None 表示全量。显存不足时用 batch_size=256 分批。
        batch 结果先在 GPU 累积，最后一次性 concat+as_host，减少 D2H 次数。
        """
        if self.model is None:
            self._parse_model()

        from .data_loader import load_input

        inputs = load_input(input_dir)
        first_key = list(inputs.keys())[0]
        total = inputs[first_key].shape[0]

        if batch_size is None or batch_size >= total:
            outputs = self._run_graph(inputs)
            return {name: as_host(arr) for name, arr in outputs.items()}

        # 在 GPU 上累积 batch 结果，最后一次性 as_host
        all_outputs = None
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = {name: arr[start:end] for name, arr in inputs.items()}
            outputs = self._run_graph(batch)
            if all_outputs is None:
                all_outputs = {name: [] for name in outputs}
            for name, arr in outputs.items():
                # 保留 device array，延后 as_host
                all_outputs[name].append(arr)

        # 最后一次 GPU concat，然后统一 as_host
        return {name: as_host(xp.concatenate(arrs, axis=0).astype(xp.float32))
                for name, arrs in all_outputs.items()}

    def run(self, input_dir, output_dir):
        if self.model is None:
            self._parse_model()

        from .data_loader import load_input
        from .data_writer import write_output

        inputs = load_input(input_dir)
        outputs = self._run_graph(inputs)
        # 输出回 host 后写盘
        host_outputs = {name: as_host(arr) for name, arr in outputs.items()}
        samples = write_output(output_dir, host_outputs)
        return samples

    def batch_run(self, input_dir, output_dir):
        if self.model is None:
            self._parse_model()

        from .data_loader import load_input
        from .data_writer import write_output

        inputs = load_input(input_dir)
        first_key = list(inputs.keys())[0]
        total_samples = inputs[first_key].shape[0]

        if self.batch_size is None or self.batch_size >= total_samples:
            return self.run(input_dir, output_dir)

        all_outputs = None

        for start in range(0, total_samples, self.batch_size):
            end = min(start + self.batch_size, total_samples)

            batch_inputs = {}
            for name, arr in inputs.items():
                batch_inputs[name] = arr[start:end]

            outputs = self._run_graph(batch_inputs)

            if all_outputs is None:
                all_outputs = {name: [] for name in outputs.keys()}

            for name, arr in outputs.items():
                all_outputs[name].append(as_host(arr))

        final_outputs = {}
        for name, arrs in all_outputs.items():
            final_outputs[name] = np.concatenate(arrs, axis=0).astype(np.float32)

        samples = write_output(output_dir, final_outputs)
        return samples


class InferenceExecutor(ONNXExecutor):
    pass
