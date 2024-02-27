# 简单的神经网络代码，不需要导入任何库
import pandas as pd
import numpy as np

# 定义 Sigmoid 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 定义神经网络类
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        
        # 初始化偏差
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))
    
    # 前向传播
    def forward(self, inputs):
        self.hidden_sum = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_activation = sigmoid(self.hidden_sum)
        
        self.output_sum = np.dot(self.hidden_activation, self.weights_hidden_output) + self.bias_output
        self.output_activation = sigmoid(self.output_sum)
        
        return self.output_activation
    
    # 反向传播
    def backward(self, inputs, targets, learning_rate):
        output_error = targets - self.output_activation
        output_delta = output_error * (self.output_activation * (1 - self.output_activation))
        
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * (self.hidden_activation * (1 - self.hidden_activation))
        
        self.weights_hidden_output += learning_rate * np.dot(self.hidden_activation.T, output_delta)
        self.bias_output += learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        self.weights_input_hidden += learning_rate * np.dot(inputs.T, hidden_delta)
        self.bias_hidden += learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

# 创建神经网络实例
input_size = 2
hidden_size = 3
output_size = 1
learning_rate = 0.1

# 定义训练数据
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

# 创建神经网络实例
nn = SimpleNeuralNetwork(input_size, hidden_size, output_size)

# 训练神经网络
epochs = 10000
for epoch in range(epochs):
    output = nn.forward(inputs)
    nn.backward(inputs, targets, learning_rate)

# 输出训练结果
print("训练后的输出：")
for i in range(len(inputs)):
    print(f"输入: {inputs[i]}, 实际输出: {output[i]}, 目标输出: {targets[i]}")

