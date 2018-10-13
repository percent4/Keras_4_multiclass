import h5py
import numpy as np

# 模型地址
MODEL_PATH = 'E://logs/iris_model.h5'

# 获取每一层的连接权重及偏重
print("读取模型中...")
with h5py.File(MODEL_PATH, 'r') as f:
    dense_1 = f['/model_weights/dense_1/dense_1']
    dense_1_bias =  dense_1['bias:0'][:]
    dense_1_kernel = dense_1['kernel:0'][:]

    dense_2 = f['/model_weights/dense_2/dense_2']
    dense_2_bias = dense_2['bias:0'][:]
    dense_2_kernel = dense_2['kernel:0'][:]

    dense_3 = f['/model_weights/dense_3/dense_3']
    dense_3_bias = dense_3['bias:0'][:]
    dense_3_kernel = dense_3['kernel:0'][:]

# 模拟每个神经层的计算，得到该层的输出
def layer_output(input, kernel, bias):
    return np.dot(input, kernel) + bias

# 实现ReLU函数
relu = np.vectorize(lambda x: x if x >=0 else 0)

# 实现softmax函数
def softmax_func(arr):
    exp_arr = np.exp(arr)
    arr_sum = np.sum(exp_arr)
    softmax_arr = exp_arr/arr_sum
    return softmax_arr

# 输入向量
unkown = np.array([[6.1, 3.1, 5.1, 1.1]], dtype=np.float32)

# 第一层的输出
print("模型计算中...")
output_1 = layer_output(unkown, dense_1_kernel, dense_1_bias)
output_1 = relu(output_1)

# 第二层的输出
output_2 = layer_output(output_1, dense_2_kernel, dense_2_bias)
output_2 = relu(output_2)

# 第三层的输出
output_3 = layer_output(output_2, dense_3_kernel, dense_3_bias)
output_3 = softmax_func(output_3)

# 最终的输出的softmax值
np.set_printoptions(precision=4)
print("最终的预测值向量为: %s"%output_3)