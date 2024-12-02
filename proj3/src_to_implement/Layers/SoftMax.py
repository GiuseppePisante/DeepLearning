from .Base import BaseLayer
import numpy as np

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        # Shift input tensor for numerical stability
        shifted_input = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
        exp_values = np.exp(shifted_input)
        self.output_tensor = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output_tensor

    def backward(self, error_tensor):
        # Use the output tensor from the forward pass
        return self.output_tensor * (error_tensor - np.sum(error_tensor * self.output_tensor, axis=1, keepdims=True))