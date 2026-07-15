#!/usr/bin/env python3
import argparse
import json
import onnx

def dtype_to_str(dtype):
    dtype_map = {
        onnx.TensorProto.FLOAT: "FLOAT",
        onnx.TensorProto.FLOAT16: "FLOAT16",
        onnx.TensorProto.DOUBLE: "DOUBLE",
        onnx.TensorProto.INT32: "INT32",
        onnx.TensorProto.INT64: "INT64",
        onnx.TensorProto.UINT32: "UINT32",
        onnx.TensorProto.UINT64: "UINT64",
        onnx.TensorProto.BOOL: "BOOL",
    }
    return dtype_map.get(dtype, "UNKNOWN")

def shape_to_list(shape):
    result = []
    for dim in shape.dim:
        if dim.dim_param:
            result.append(dim.dim_param)
        elif dim.dim_value != 0:
            result.append(int(dim.dim_value))
        else:
            result.append("?")
    return result

def export_dag(onnx_path, output_path):
    model = onnx.load(onnx_path)
    graph = model.graph
    
    initializer_names = set([init.name for init in graph.initializer])
    
    graph_inputs = []
    for inp in graph.input:
        if inp.name not in initializer_names:
            dtype = dtype_to_str(inp.type.tensor_type.elem_type)
            shape = shape_to_list(inp.type.tensor_type.shape)
            graph_inputs.append({
                "name": inp.name,
                "dtype": dtype,
                "shape": shape
            })
    
    graph_outputs = []
    for out in graph.output:
        dtype = dtype_to_str(out.type.tensor_type.elem_type)
        shape = shape_to_list(out.type.tensor_type.shape)
        graph_outputs.append({
            "name": out.name,
            "dtype": dtype,
            "shape": shape
        })
    
    nodes = []
    node_outputs = {}
    for node in graph.node:
        node_info = {
            "name": node.name,
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output)
        }
        nodes.append(node_info)
        for output_name in node.output:
            node_outputs[output_name] = node.name
    
    edges = []
    for node in graph.node:
        for input_name in node.input:
            if input_name in node_outputs:
                edges.append({
                    "src_node": node_outputs[input_name],
                    "dst_node": node.name,
                    "tensor": input_name
                })
    
    dag = {
        "format_version": "1.0",
        "graph_inputs": graph_inputs,
        "graph_outputs": graph_outputs,
        "nodes": nodes,
        "edges": edges
    }
    
    with open(output_path, "w") as f:
        json.dump(dag, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Export ONNX model to DAG JSON")
    parser.add_argument("--onnx", required=True, help="Input ONNX model path")
    parser.add_argument("--output", required=True, help="Output DAG JSON path")
    args = parser.parse_args()
    
    export_dag(args.onnx, args.output)

if __name__ == "__main__":
    main()
