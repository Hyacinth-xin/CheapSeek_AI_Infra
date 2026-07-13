import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph import import_onnx_graph
from graph_passes import FusionPass

print('=== 测试 fusion_test.onnx ===')
g = import_onnx_graph('models/fusion_test.onnx')
print('融合前节点数:', len(g.nodes))
for node in g.nodes:
    print('  ', node.name, ':', node.op_type)

fusion = FusionPass()
g_fused = fusion.apply(g)

print('\n融合后节点数:', len(g_fused.nodes))
for node in g_fused.nodes:
    print('  ', node.name, ':', node.op_type)

print('\n融合统计:')
stats = fusion.get_stats()
print('  融合节点数:', stats['fused_count'])
print('  模式覆盖率:', stats['pattern_coverage'])
for log in stats['fusion_log']:
    print('  -', log['pattern'], ':', log['nodes'], '->', log['result'])
