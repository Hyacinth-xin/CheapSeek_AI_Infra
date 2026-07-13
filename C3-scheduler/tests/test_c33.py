import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph import import_onnx_graph
from graph_passes import FusionPass

print('=== 测试 transformer_v1.onnx ===')
g = import_onnx_graph('models/transformer_v1.onnx')
original_count = len(g.nodes)
print('融合前节点数:', original_count)

fusion = FusionPass()
g_fused = fusion.apply(g)

fused_count = len(g_fused.nodes)
print('融合后节点数:', fused_count)
reduction_rate = (original_count - fused_count) / original_count * 100
print('融合减少率: {:.1f}%'.format(reduction_rate))

print('\n融合统计:')
stats = fusion.get_stats()
print('  融合节点数:', stats['fused_count'])
print('  模式覆盖率:', stats['pattern_coverage'])
for log in stats['fusion_log']:
    print('  -', log['pattern'], ':', log['nodes'], '->', log['result'])