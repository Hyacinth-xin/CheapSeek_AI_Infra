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

        graph = self._fuse_matmul_bias(graph)
        graph = self._fuse_conv_bn(graph)
        graph = self._fuse_ew_chain(graph)
        graph = self._fuse_softmax_dropout(graph)
        graph = self._fuse_residual_norm(graph)

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
        new_nodes = []
        i = 0
        while i < len(graph.nodes):
            node = graph.nodes[i]
            if node.op_type in {"Add", "Mul", "Relu", "Div", "Erf", "Sub"}:
                chain = [node]
                j = i + 1
                while j < len(graph.nodes) and len(chain) < 5:
                    next_node = graph.nodes[j]
                    if next_node.op_type in {"Add", "Mul", "Relu", "Div", "Erf", "Sub"}:
                        if chain[-1].outputs[0] in next_node.inputs:
                            chain.append(next_node)
                            j += 1
                        else:
                            break
                    else:
                        break
                
                if len(chain) >= 2:
                    all_inputs = []
                    outputs = chain[-1].outputs
                    seen = set()
                    for n in chain:
                        for inp in n.inputs:
                            if inp not in seen:
                                seen.add(inp)
                                all_inputs.append(inp)
                    
                    new_node = NodeInfo(
                        name=f"{chain[0].name}_fused_chain",
                        op_type="FusedEWChain",
                        inputs=all_inputs,
                        outputs=outputs,
                        attrs={"original_ops": [n.op_type for n in chain]}
                    )
                    new_nodes.append(new_node)
                    self.fusion_log.append({
                        "pattern": "FusedEWChain",
                        "nodes": [n.name for n in chain],
                        "result": new_node.name
                    })
                    self.fused_count += len(chain) - 1
                    i = j
                    continue
            new_nodes.append(node)
            i += 1
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
