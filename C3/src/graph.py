from typing import List, Dict, Optional
from dataclasses import dataclass, field
import onnx


@dataclass
class TensorInfo:
    name: str
    dtype: str
    shape: List
    is_initializer: bool = False
    data: Optional[bytes] = None


@dataclass
class NodeInfo:
    name: str
    op_type: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    attrs: Dict = field(default_factory=dict)


class Graph:
    def __init__(self):
        self.nodes: List[NodeInfo] = []
        self.tensors: Dict[str, TensorInfo] = {}
        self.inputs: List[str] = []
        self.outputs: List[str] = []
        self.node_map: Dict[str, NodeInfo] = {}

    def add_node(self, node: NodeInfo):
        self.nodes.append(node)
        self.node_map[node.name] = node

    def add_tensor(self, tensor: TensorInfo):
        self.tensors[tensor.name] = tensor

    def get_node(self, name: str) -> Optional[NodeInfo]:
        return self.node_map.get(name)

    def get_tensor(self, name: str) -> Optional[TensorInfo]:
        return self.tensors.get(name)

    def validate(self) -> bool:
        return True


def import_onnx_graph(onnx_path: str) -> Graph:
    model = onnx.load(onnx_path)
    graph = Graph()

    initializer_names = set([init.name for init in model.graph.initializer])

    for init in model.graph.initializer:
        dtype_map = {
            onnx.TensorProto.FLOAT: "FLOAT",
            onnx.TensorProto.FLOAT16: "FLOAT16",
            onnx.TensorProto.INT32: "INT32",
            onnx.TensorProto.INT64: "INT64",
        }
        dtype = dtype_map.get(init.data_type, "UNKNOWN")
        shape = []
        for dim in init.dims:
            shape.append(dim)
        graph.add_tensor(TensorInfo(
            name=init.name,
            dtype=dtype,
            shape=shape,
            is_initializer=True
        ))

    for inp in model.graph.input:
        if inp.name not in initializer_names:
            dtype_map = {
                onnx.TensorProto.FLOAT: "FLOAT",
                onnx.TensorProto.FLOAT16: "FLOAT16",
                onnx.TensorProto.INT32: "INT32",
                onnx.TensorProto.INT64: "INT64",
            }
            dtype = dtype_map.get(inp.type.tensor_type.elem_type, "UNKNOWN")
            shape = []
            for dim in inp.type.tensor_type.shape.dim:
                if dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    shape.append(dim.dim_value)
            graph.add_tensor(TensorInfo(
                name=inp.name,
                dtype=dtype,
                shape=shape,
                is_initializer=False
            ))
            graph.inputs.append(inp.name)

    for out in model.graph.output:
        graph.outputs.append(out.name)

    for node in model.graph.node:
        node_info = NodeInfo(
            name=node.name,
            op_type=node.op_type,
            inputs=list(node.input),
            outputs=list(node.output)
        )
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.FLOAT:
                node_info.attrs[attr.name] = attr.f
            elif attr.type == onnx.AttributeProto.INT:
                node_info.attrs[attr.name] = attr.i
            elif attr.type == onnx.AttributeProto.STRING:
                node_info.attrs[attr.name] = attr.s.decode("utf-8")
            elif attr.type == onnx.AttributeProto.INTS:
                node_info.attrs[attr.name] = list(attr.ints)
            elif attr.type == onnx.AttributeProto.FLOATS:
                node_info.attrs[attr.name] = list(attr.floats)
        graph.add_node(node_info)

    return graph
