import onnx
from onnx import helper, TensorProto
import numpy as np

X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3, 28, 28])
W1 = helper.make_tensor_value_info('W1', TensorProto.FLOAT, [32, 3, 3, 3])
B1 = helper.make_tensor_value_info('B1', TensorProto.FLOAT, [32])
W2 = helper.make_tensor_value_info('W2', TensorProto.FLOAT, [64, 32])
B2 = helper.make_tensor_value_info('B2', TensorProto.FLOAT, [64])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 64])

conv_node = helper.make_node('Conv', ['X', 'W1', 'B1'], ['conv_out'])
bn_node = helper.make_node('BatchNormalization', ['conv_out', 'bn_scale', 'bn_bias', 'bn_mean', 'bn_var'], ['bn_out'], epsilon=1e-5)
relu_node = helper.make_node('Relu', ['bn_out'], ['relu_out'])
matmul_node = helper.make_node('MatMul', ['relu_out', 'W2'], ['matmul_out'])
add_node = helper.make_node('Add', ['matmul_out', 'B2'], ['add_out'])
softmax_node = helper.make_node('Softmax', ['add_out'], ['softmax_out'])

bn_scale = helper.make_tensor('bn_scale', TensorProto.FLOAT, [32], np.ones(32).astype(np.float32))
bn_bias = helper.make_tensor('bn_bias', TensorProto.FLOAT, [32], np.zeros(32).astype(np.float32))
bn_mean = helper.make_tensor('bn_mean', TensorProto.FLOAT, [32], np.zeros(32).astype(np.float32))
bn_var = helper.make_tensor('bn_var', TensorProto.FLOAT, [32], np.ones(32).astype(np.float32))

graph = helper.make_graph(
    [conv_node, bn_node, relu_node, matmul_node, add_node, softmax_node],
    'fusion_test_graph',
    [X, W1, B1, W2, B2],
    [Y],
    [bn_scale, bn_bias, bn_mean, bn_var]
)

model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
onnx.save(model, 'C3-scheduler/models/fusion_test.onnx')
print('Created fusion_test.onnx')
