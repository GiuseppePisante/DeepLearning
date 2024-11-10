from .Base import BaseLayer
import numpy as np

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        # Add bias term to input tensor
        self.input_tensor = input_tensor
        output = np.maximum(0, input_tensor)
        return output.copy()
    
    def backward(self, error_tensor):
        # f'(x) = 1 if x > 0, 0 otherwise
        error_tensor[self.input_tensor <= 0] = 0
        return error_tensor



