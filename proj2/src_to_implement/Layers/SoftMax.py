from .Base import BaseLayer
import numpy as np

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        # Add bias term to input tensor
        self.input_tensor = input_tensor
        input_exp = np.exp(self.input_tensor)
        output = input_exp / np.sum(input_exp, axis=1, keepdims=True)
        return output.copy()
    
    def backward(self, error_tensor):
        


        return error_tensor