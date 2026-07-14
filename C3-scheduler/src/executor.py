import numpy as np
import onnx
from scipy import signal
from pathlib import Path


class ONNXExecutor:
    def __init__(self, onnx_path, batch_size=None):
        self.onnx_path = Path(onnx_path)
        self.batch_size = batch_size
        self.model = None
        self.graph = None
        self.input_names = None
        self.output_names = None
        self.value_info = {}
        self.initializers = {}
    
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
        
        for init in self.graph.initializer:
            arr = onnx.numpy_helper.to_array(init)
            self.initializers[init.name] = arr.astype(np.float32)
    
    def _execute_node(self, node, tensors):
        op_type = node.op_type
        
        inputs = []
        for inp_name in node.input:
            if inp_name in tensors:
                inputs.append(tensors[inp_name])
            elif inp_name in self.initializers:
                inputs.append(self.initializers[inp_name])
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
            result = np.maximum(inputs[0], 0)
        elif op_type == "Erf":
            from scipy.special import erf
            result = erf(inputs[0])
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
            exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
            result = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        elif op_type == "LayerNormalization":
            x, scale, bias = inputs[0], inputs[1], inputs[2]
            eps = 1e-5
            for attr in node.attribute:
                if attr.name == "epsilon":
                    eps = attr.f
            mean = np.mean(x, axis=-1, keepdims=True)
            var = np.var(x, axis=-1, keepdims=True)
            norm = (x - mean) / np.sqrt(var + eps)
            result = norm * scale + bias
        elif op_type == "Reshape":
            x, shape_tensor = inputs[0], inputs[1]
            if shape_tensor is None:
                shape = []
                for attr in node.attribute:
                    if attr.name == "shape":
                        shape = list(attr.ints)
            elif isinstance(shape_tensor, list):
                shape = shape_tensor
            elif isinstance(shape_tensor, np.ndarray):
                shape = shape_tensor.astype(np.int64).tolist()
            else:
                shape = list(shape_tensor)
            
            input_shape = x.shape
            new_shape = []
            for dim in shape:
                if dim == 0:
                    new_shape.append(input_shape[len(new_shape)])
                elif dim == -1:
                    remaining = 1
                    for i in range(len(new_shape), len(input_shape)):
                        remaining *= input_shape[i]
                    for d in shape:
                        if d > 0:
                            remaining //= d
                    new_shape.append(remaining)
                else:
                    new_shape.append(dim)
            
            result = x.reshape(new_shape)
        elif op_type == "Transpose":
            x = inputs[0]
            perm = list(range(x.ndim))
            for attr in node.attribute:
                if attr.name == "perm":
                    perm = list(attr.ints)
            result = np.transpose(x, perm)
        elif op_type == "Flatten":
            x = inputs[0]
            axis = 1
            for attr in node.attribute:
                if attr.name == "axis":
                    axis = attr.i
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
            
            result = np.split(x, np.cumsum(split[:-1]), axis=axis)
        elif op_type == "Gather":
            x, indices = inputs[0], inputs[1]
            axis = 0
            for attr in node.attribute:
                if attr.name == "axis":
                    axis = attr.i
            result = np.take(x, indices.astype(np.int64), axis=axis)
        elif op_type == "Constant":
            value = None
            for attr in node.attribute:
                if attr.name == "value":
                    arr = onnx.numpy_helper.to_array(attr.t)
                    value = arr.astype(np.float32)
            result = value
        elif op_type == "Conv":
            x, w, b = inputs[0], inputs[1], inputs[2] if len(inputs) > 2 else None
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
            
            N, C_in, H_in, W_in = x.shape
            C_out, C_in_g, KH, KW = w.shape
            
            out_h = (H_in + padding[0] + padding[2] - dilation[0] * (KH - 1) - 1) // stride[0] + 1
            out_w = (W_in + padding[1] + padding[3] - dilation[1] * (KW - 1) - 1) // stride[1] + 1
            
            x_padded = np.zeros((N, C_in, H_in + padding[0] + padding[2], W_in + padding[1] + padding[3]), dtype=np.float32)
            x_padded[:, :, padding[0]:padding[0]+H_in, padding[1]:padding[1]+W_in] = x
            
            result = np.zeros((N, C_out, out_h, out_w), dtype=np.float32)
            
            for n in range(N):
                for c_out in range(C_out):
                    for c_in in range(C_in_g):
                        img = x_padded[n, c_in + (c_out // (C_out // groups)) * C_in_g, :, :]
                        kernel = w[c_out, c_in, ::-1, ::-1]
                        conv_out = signal.convolve2d(img, kernel, mode='valid')
                        result[n, c_out, :, :] += conv_out[::stride[0], ::stride[1]]
            
            if b is not None:
                result = result + b.reshape(1, -1, 1, 1)
        elif op_type == "GlobalAveragePool":
            x = inputs[0]
            result = np.mean(x, axis=(2, 3))
        else:
            raise NotImplementedError(f"Unsupported operator: {op_type}")
        
        return result
    
    def _run_graph(self, input_tensors):
        tensors = {}
        for name, arr in input_tensors.items():
            if arr.dtype == np.int64:
                tensors[name] = arr.astype(np.int64)
            else:
                tensors[name] = arr.astype(np.float32)
        
        for node in self.graph.node:
            result = self._execute_node(node, tensors)
            
            if isinstance(result, (tuple, list)):
                for i, out_name in enumerate(node.output):
                    tensors[out_name] = result[i]
            else:
                for out_name in node.output:
                    tensors[out_name] = result
        
        outputs = {}
        for name in self.output_names:
            if name in tensors:
                outputs[name] = tensors[name].astype(np.float32)
        
        return outputs
    
    def run(self, input_dir, output_dir):
        if self.model is None:
            self._parse_model()
        
        from .data_loader import load_input
        from .data_writer import write_output
        
        inputs = load_input(input_dir)
        outputs = self._run_graph(inputs)
        samples = write_output(output_dir, outputs)
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
                all_outputs[name].append(arr)
        
        final_outputs = {}
        for name, arrs in all_outputs.items():
            final_outputs[name] = np.concatenate(arrs, axis=0).astype(np.float32)
        
        samples = write_output(output_dir, final_outputs)
        return samples


class InferenceExecutor(ONNXExecutor):
    pass
