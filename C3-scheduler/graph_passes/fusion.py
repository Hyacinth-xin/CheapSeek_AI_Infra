from typing import List, Dict, Optional, Tuple
from graph import Graph, NodeInfo


class FusionPass:
    def __init__(self):
        self.fusion_log = []
        self.fused_count = 0

    def apply(self, graph: Graph) -> Graph:
        self.fusion_log = []
        self.fused_count = 0

        # 记录原始输出张量
        original_outputs = set(graph.outputs)
        original_inputs = set(graph.inputs)

        graph = self._fuse_split_reshape(graph)
        graph = self._fuse_reshape_transpose(graph)
        graph = self._fuse_matmul_bias(graph)
        graph = self._fuse_matmul_transpose(graph)
        graph = self._fuse_transpose_matmul(graph)
        graph = self._fuse_conv_bn(graph)
        graph = self._fuse_softmax_dropout(graph)
        graph = self._fuse_residual_norm(graph)
        graph = self._fuse_ew_chain(graph)

        # 更新 outputs：确保原始输出张量仍然存在
        # 检查哪些原始输出被融合了，更新为融合后的输出
        updated_outputs = []
        for orig_out in original_outputs:
            found = False
            for node in graph.nodes:
                if orig_out in node.outputs:
                    updated_outputs.append(orig_out)
                    found = True
                    break
            if not found:
                # 输出被融合了，使用融合节点的输出
                for node in graph.nodes:
                    if node.outputs:
                        updated_outputs.append(node.outputs[0])
                        break
        graph.outputs = list(set(updated_outputs)) if updated_outputs else []

        # 更新 inputs：确保原始输入张量仍然存在
        updated_inputs = []
        for orig_in in original_inputs:
            found = False
            for node in graph.nodes:
                if orig_in in node.inputs:
                    updated_inputs.append(orig_in)
                    found = True
                    break
            if not found:
                # 输入被使用但可能在某个融合节点中
                updated_inputs.append(orig_in)
        graph.inputs = list(set(updated_inputs)) if updated_inputs else []

        # 更新 node_map
        graph.node_map = {n.name: n for n in graph.nodes}

        return graph

    def _fuse_matmul_bias(self, graph: Graph) -> Graph:
        new_nodes = []
        i = 0
        while i < len(graph.nodes):
            node = graph.nodes[i]
            if node.op_type in {"Gemm", "MatMul"}:
                if i + 1 < len(graph.nodes):
                    next_node = graph.nodes[i + 1]
                    if next_node.op_type == "Add" and len(node.outputs) == 1:
                        if node.outputs[0] in next_node.inputs:
                            new_node = NodeInfo(
                                name=f"{node.name}_fused_bias",
                                op_type="FusedMatMulBias",
                                inputs=node.inputs + [next_node.inputs[0] if next_node.inputs[1] == node.outputs[0] else next_node.inputs[1]],
                                outputs=next_node.outputs,
                                attrs={"original_ops": [node.op_type, "Add"]}
                            )
                            new_nodes.append(new_node)
                            self.fusion_log.append({
                                "pattern": "FusedMatMulBias",
                                "nodes": [node.name, next_node.name],
                                "result": new_node.name
                            })
                            self.fused_count += 1
                            i += 2
                            continue
            new_nodes.append(node)
            i += 1
        graph.nodes = new_nodes
        graph.node_map = {n.name: n for n in graph.nodes}
        return graph

    def _fuse_conv_bn(self, graph: Graph) -> Graph:
        new_nodes = []
        i = 0
        while i < len(graph.nodes):
            node = graph.nodes[i]
            if node.op_type == "Conv":
                if i + 1 < len(graph.nodes):
                    next_node = graph.nodes[i + 1]
                    if next_node.op_type == "BatchNormalization" and len(node.outputs) == 1:
                        if node.outputs[0] in next_node.inputs:
                            new_node = NodeInfo(
                                name=f"{node.name}_fused_bn",
                                op_type="FusedConv2dBatchNorm",
                                inputs=node.inputs + [next_node.inputs[1], next_node.inputs[2], 
                                                      next_node.inputs[3], next_node.inputs[4]],
                                outputs=next_node.outputs,
                                attrs={"original_ops": ["Conv", "BatchNormalization"]}
                            )
                            new_nodes.append(new_node)
                            self.fusion_log.append({
                                "pattern": "FusedConv2dBatchNorm",
                                "nodes": [node.name, next_node.name],
                                "result": new_node.name
                            })
                            self.fused_count += 1
                            i += 2
                            continue
            new_nodes.append(node)
            i += 1
        graph.nodes = new_nodes
        graph.node_map = {n.name: n for n in graph.nodes}
        return graph

    def _fuse_ew_chain(self, graph: Graph) -> Graph:
        ew_ops = {"Add", "Mul", "Relu", "Div", "Erf", "Sub", "Softmax", "MatMul"}
        
        input_map = {}
        for node in graph.nodes:
            for inp in node.inputs:
                if inp not in input_map:
                    input_map[inp] = []
                input_map[inp].append(node)
        
        visited = set()
        new_nodes = []
        
        for node in graph.nodes:
            if node.name in visited:
                continue
            
            if node.op_type not in ew_ops:
                new_nodes.append(node)
                continue
            
            chain = []
            current_node = node
            
            while current_node is not None and current_node.op_type in ew_ops:
                if current_node.name in visited:
                    break
                
                visited.add(current_node.name)
                chain.append(current_node)
                
                if not current_node.outputs:
                    break
                
                chain_output = current_node.outputs[0]
                
                consumers = []
                if chain_output in input_map:
                    for n in input_map[chain_output]:
                        if n.name not in visited and n.op_type in ew_ops:
                            consumers.append(n)
                
                if len(consumers) == 1:
                    current_node = consumers[0]
                else:
                    break
            
            if len(chain) >= 2:
                all_inputs = []
                seen = set()
                for n in chain:
                    for inp in n.inputs:
                        if inp not in seen:
                            seen.add(inp)
                            all_inputs.append(inp)
                
                outputs = chain[-1].outputs
                
                op_types = [n.op_type for n in chain]
                if "MatMul" in op_types:
                    op_type = "FusedEWMatMulChain"
                elif "Softmax" in op_types:
                    op_type = "FusedEWSoftmaxChain"
                else:
                    op_type = "FusedEWChain"
                
                new_node = NodeInfo(
                    name=f"{chain[0].name}_fused_chain",
                    op_type=op_type,
                    inputs=all_inputs,
                    outputs=outputs,
                    attrs={"original_ops": op_types}
                )
                new_nodes.append(new_node)
                self.fusion_log.append({
                    "pattern": op_type,
                    "nodes": [n.name for n in chain],
                    "result": new_node.name
                })
                self.fused_count += len(chain) - 1
            elif chain:
                for n in chain:
                    visited.remove(n.name)
                new_nodes.append(node)
        
        graph.nodes = new_nodes
        graph.node_map = {n.name: n for n in graph.nodes}
        return graph

    def _fuse_reshape_transpose(self, graph: Graph) -> Graph:
        new_nodes = []
        i = 0
        while i < len(graph.nodes):
            node = graph.nodes[i]
            if node.op_type in {"Reshape", "Transpose"}:
                if i + 1 < len(graph.nodes):
                    next_node = graph.nodes[i + 1]
                    if next_node.op_type in {"Reshape", "Transpose"} and len(node.outputs) == 1:
                        if node.outputs[0] in next_node.inputs:
                            new_node = NodeInfo(
                                name=f"{node.name}_fused_rt",
                                op_type="FusedReshapeTranspose",
                                inputs=node.inputs + next_node.inputs[1:],
                                outputs=next_node.outputs,
                                attrs={"original_ops": [node.op_type, next_node.op_type]}
                            )
                            new_nodes.append(new_node)
                            self.fusion_log.append({
                                "pattern": "FusedReshapeTranspose",
                                "nodes": [node.name, next_node.name],
                                "result": new_node.name
                            })
                            self.fused_count += 1
                            i += 2
                            continue
            new_nodes.append(node)
            i += 1
        graph.nodes = new_nodes
        graph.node_map = {n.name: n for n in graph.nodes}
        return graph

    def _fuse_matmul_transpose(self, graph: Graph) -> Graph:
        input_map = {}
        for node in graph.nodes:
            for inp in node.inputs:
                if inp not in input_map:
                    input_map[inp] = []
                input_map[inp].append(node)
        
        visited = set()
        new_nodes = []
        
        for node in graph.nodes:
            if node.name in visited:
                continue
            
            if node.op_type != "MatMul":
                new_nodes.append(node)
                continue
            
            if not node.outputs:
                new_nodes.append(node)
                continue
            
            chain_output = node.outputs[0]
            if chain_output in input_map:
                consumers = input_map[chain_output]
                for consumer in consumers:
                    if consumer.op_type == "Transpose" and consumer.name not in visited:
                        visited.add(node.name)
                        visited.add(consumer.name)
                        
                        all_inputs = list(node.inputs)
                        for inp in consumer.inputs:
                            if inp != chain_output and inp not in all_inputs:
                                all_inputs.append(inp)
                        
                        new_node = NodeInfo(
                            name=f"{node.name}_fused_transpose",
                            op_type="FusedMatMulTranspose",
                            inputs=all_inputs,
                            outputs=consumer.outputs,
                            attrs={"original_ops": ["MatMul", "Transpose"]}
                        )
                        new_nodes.append(new_node)
                        self.fusion_log.append({
                            "pattern": "FusedMatMulTranspose",
                            "nodes": [node.name, consumer.name],
                            "result": new_node.name
                        })
                        self.fused_count += 1
                        break
                else:
                    new_nodes.append(node)
            else:
                new_nodes.append(node)
        
        graph.nodes = new_nodes
        graph.node_map = {n.name: n for n in graph.nodes}
        return graph

    def _fuse_transpose_matmul(self, graph: Graph) -> Graph:
        input_map = {}
        for node in graph.nodes:
            for inp in node.inputs:
                if inp not in input_map:
                    input_map[inp] = []
                input_map[inp].append(node)
        
        visited = set()
        new_nodes = []
        
        for node in graph.nodes:
            if node.name in visited:
                continue
            
            if node.op_type != "Transpose":
                new_nodes.append(node)
                continue
            
            if not node.outputs:
                new_nodes.append(node)
                continue
            
            chain_output = node.outputs[0]
            if chain_output in input_map:
                consumers = input_map[chain_output]
                for consumer in consumers:
                    if consumer.op_type == "MatMul" and consumer.name not in visited:
                        visited.add(node.name)
                        visited.add(consumer.name)
                        
                        all_inputs = list(node.inputs)
                        for inp in consumer.inputs:
                            if inp != chain_output and inp not in all_inputs:
                                all_inputs.append(inp)
                        
                        new_node = NodeInfo(
                            name=f"{node.name}_fused_matmul",
                            op_type="FusedTransposeMatMul",
                            inputs=all_inputs,
                            outputs=consumer.outputs,
                            attrs={"original_ops": ["Transpose", "MatMul"]}
                        )
                        new_nodes.append(new_node)
                        self.fusion_log.append({
                            "pattern": "FusedTransposeMatMul",
                            "nodes": [node.name, consumer.name],
                            "result": new_node.name
                        })
                        self.fused_count += 1
                        break
                else:
                    new_nodes.append(node)
            else:
                new_nodes.append(node)
        
        graph.nodes = new_nodes
        graph.node_map = {n.name: n for n in graph.nodes}
        return graph

    def _fuse_split_reshape(self, graph: Graph) -> Graph:
        input_map = {}
        for node in graph.nodes:
            for inp in node.inputs:
                if inp not in input_map:
                    input_map[inp] = []
                input_map[inp].append(node)
        
        visited = set()
        new_nodes = []
        
        for node in graph.nodes:
            if node.name in visited:
                continue
            
            if node.op_type != "Split":
                new_nodes.append(node)
                continue
            
            reshapes = []
            for out_tensor in node.outputs:
                if out_tensor in input_map:
                    consumers = input_map[out_tensor]
                    for consumer in consumers:
                        if consumer.op_type == "Reshape" and consumer.name not in visited:
                            reshapes.append(consumer)
            
            if len(reshapes) >= 2:
                visited.add(node.name)
                for r in reshapes:
                    visited.add(r.name)
                
                all_inputs = list(node.inputs)
                for r in reshapes:
                    for inp in r.inputs:
                        if inp not in node.outputs and inp not in all_inputs:
                            all_inputs.append(inp)
                
                new_node = NodeInfo(
                    name=f"{node.name}_fused_split_reshape",
                    op_type="FusedSplitReshape",
                    inputs=all_inputs,
                    outputs=[r.outputs[0] for r in reshapes],
                    attrs={"original_ops": ["Split"] + ["Reshape"] * len(reshapes)}
                )
                new_nodes.append(new_node)
                self.fusion_log.append({
                    "pattern": "FusedSplitReshape",
                    "nodes": [node.name] + [r.name for r in reshapes],
                    "result": new_node.name
                })
                self.fused_count += len(reshapes)
            else:
                new_nodes.append(node)
        
        graph.nodes = new_nodes
        graph.node_map = {n.name: n for n in graph.nodes}
        return graph

    def _fuse_softmax_dropout(self, graph: Graph) -> Graph:
        new_nodes = []
        i = 0
        while i < len(graph.nodes):
            node = graph.nodes[i]
            if node.op_type == "Softmax":
                if i + 1 < len(graph.nodes):
                    next_node = graph.nodes[i + 1]
                    if next_node.op_type == "Dropout" and len(node.outputs) == 1:
                        if node.outputs[0] in next_node.inputs:
                            new_node = NodeInfo(
                                name=f"{node.name}_fused_dropout",
                                op_type="FusedSoftmaxDropout",
                                inputs=node.inputs + [next_node.inputs[1], next_node.inputs[2]],
                                outputs=next_node.outputs,
                                attrs={"original_ops": ["Softmax", "Dropout"]}
                            )
                            new_nodes.append(new_node)
                            self.fusion_log.append({
                                "pattern": "FusedSoftmaxDropout",
                                "nodes": [node.name, next_node.name],
                                "result": new_node.name
                            })
                            self.fused_count += 1
                            i += 2
                            continue
            new_nodes.append(node)
            i += 1
        graph.nodes = new_nodes
        graph.node_map = {n.name: n for n in graph.nodes}
        return graph

    def _fuse_residual_norm(self, graph: Graph) -> Graph:
        new_nodes = []
        i = 0
        while i < len(graph.nodes):
            node = graph.nodes[i]
            if node.op_type == "Add":
                if i + 1 < len(graph.nodes):
                    next_node = graph.nodes[i + 1]
                    if next_node.op_type == "LayerNormalization" and len(node.outputs) == 1:
                        if node.outputs[0] in next_node.inputs:
                            new_node = NodeInfo(
                                name=f"{node.name}_fused_norm",
                                op_type="FusedResidualNorm",
                                inputs=node.inputs + next_node.inputs[1:],
                                outputs=next_node.outputs,
                                attrs={"original_ops": ["Add", "LayerNormalization"]}
                            )
                            new_nodes.append(new_node)
                            self.fusion_log.append({
                                "pattern": "FusedResidualNorm",
                                "nodes": [node.name, next_node.name],
                                "result": new_node.name
                            })
                            self.fused_count += 1
                            i += 2
                            continue
            new_nodes.append(node)
            i += 1
        graph.nodes = new_nodes
        graph.node_map = {n.name: n for n in graph.nodes}
        return graph

    def get_stats(self) -> Dict:
        return {
            "fused_count": self.fused_count,
            "fusion_log": self.fusion_log,
            "pattern_coverage": list(set([f["pattern"] for f in self.fusion_log]))
        }


class GraphPassPipeline:
    def __init__(self, enable_fusion=True):
        self.enable_fusion = enable_fusion
        self.pass_results = {}

    def run(self, graph: Graph) -> Graph:
        if self.enable_fusion:
            fusion_pass = FusionPass()
            graph = fusion_pass.apply(graph)
            self.pass_results["Fusion"] = {
                "stats": fusion_pass.get_stats()
            }
        return graph
