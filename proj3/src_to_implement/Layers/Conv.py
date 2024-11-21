from .Base import BaseLayer
import numpy as np
from scipy.ndimage import correlate, convolve

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        if len(convolution_shape) == 2:  # 1D Convolution
            c, m = convolution_shape
            self.weights = np.random.uniform(0, 1, size=(num_kernels, c, m))
        elif len(convolution_shape) == 3:  # 2D Convolution
            c, m, n = convolution_shape
            self.weights = np.random.uniform(0, 1, size=(num_kernels, c, m, n))
        else:
            raise ValueError("Invalid convolution shape. Must be [c, m] or [c, m, n].")
        
        self.bias = np.random.uniform(0, 1, size=(num_kernels,))
        self.trainable = True
        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias = np.zeros_like(self.bias)
        self._optimizer_weights = None
        self._optimizer_bias = None

    @property
    def optimizer(self):
        return self._optimizer_weights, self._optimizer_bias

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer_weights = value
        self._optimizer_bias = value

    def forward(self, input_tensor):
        if len(input_tensor.shape) == 3:  # 1D Convolution
            b, c, y = input_tensor.shape
            output_shape = (b, self.num_kernels, (y - self.convolution_shape[1]) // self.stride_shape[0] + 1)
            padded_input = np.pad(input_tensor, ((0, 0), (0, 0), (self.convolution_shape[1] // 2, self.convolution_shape[1] // 2)), mode='constant')
        elif len(input_tensor.shape) == 4:  # 2D Convolution
            b, c, y, x = input_tensor.shape
            output_shape = (b, self.num_kernels, (y - self.convolution_shape[1]) // self.stride_shape[0] + 1, (x - self.convolution_shape[2]) // self.stride_shape[1] + 1)
            padded_input = np.pad(input_tensor, ((0, 0), (0, 0), (self.convolution_shape[1] // 2, self.convolution_shape[1] // 2), (self.convolution_shape[2] // 2, self.convolution_shape[2] // 2)), mode='constant')
        else:
            raise ValueError("Invalid input tensor shape. Must be [b, c, y] or [b, c, y, x].")

        output_tensor = np.zeros(output_shape)

        for i in range(self.num_kernels):
            for j in range(c):
                if len(input_tensor.shape) == 3:  # 1D Convolution
                    output_tensor[:, i, :] += correlate(padded_input[:, j, :], self.weights[i, j, :], mode='constant')
                elif len(input_tensor.shape) == 4:  # 2D Convolution
                    output_tensor[:, i, :, :] += correlate(padded_input[:, j, :, :], self.weights[i, j, :, :], mode='constant')

            output_tensor[:, i, :] += self.bias[i]

        return output_tensor

    def backward(self, error_tensor):
        if len(error_tensor.shape) == 3:  # 1D Convolution
            b, k, y = error_tensor.shape
            padded_error = np.pad(error_tensor, ((0, 0), (0, 0), (self.convolution_shape[1] // 2, self.convolution_shape[1] // 2)), mode='constant')
            gradient_input = np.zeros((b, self.weights.shape[1], y))
            for i in range(self.num_kernels):
                for j in range(self.weights.shape[1]):
                    self._gradient_weights[i, j, :] = correlate(padded_error[:, i, :], self.weights[i, j, :], mode='constant')
                    gradient_input[:, j, :] += convolve(padded_error[:, i, :], self.weights[i, j, :], mode='constant')
                self._gradient_bias[i] = np.sum(error_tensor[:, i, :], axis=(0, 1))
        elif len(error_tensor.shape) == 4:  # 2D Convolution
            b, k, y, x = error_tensor.shape
            padded_error = np.pad(error_tensor, ((0, 0), (0, 0), (self.convolution_shape[1] // 2, self.convolution_shape[1] // 2), (self.convolution_shape[2] // 2, self.convolution_shape[2] // 2)), mode='constant')
            gradient_input = np.zeros((b, self.weights.shape[1], y, x))
            for i in range(self.num_kernels):
                for j in range(self.weights.shape[1]):
                    self._gradient_weights[i, j, :, :] = correlate(padded_error[:, i, :, :], self.weights[i, j, :, :], mode='constant')
                    gradient_input[:, j, :, :] += convolve(padded_error[:, i, :, :], self.weights[i, j, :, :], mode='constant')
                self._gradient_bias[i] = np.sum(error_tensor[:, i, :, :], axis=(0, 1, 2))
        else:
            raise ValueError("Invalid error tensor shape. Must be [b, k, y] or [b, k, y, x].")

        if self._optimizer_weights:
            self.weights = self._optimizer_weights.calculate_update(self.weights, self._gradient_weights)
        if self._optimizer_bias:
            self.bias = self._optimizer_bias.calculate_update(self.bias, self._gradient_bias)

        return gradient_input

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape, np.prod(self.convolution_shape), self.num_kernels)
        self.bias = bias_initializer.initialize(self.bias.shape, 1, self.num_kernels)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias