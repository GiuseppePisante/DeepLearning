from .Base import BaseLayer
import numpy as np

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.weights = np.random.rand(input_size + 1, output_size)
        
    def forward(self, input_tensor):
        # Add bias term to input tensor
        bias = np.ones((input_tensor.shape[0], 1))
        input_tensor = np.hstack((input_tensor, bias))
        self.input_tensor = input_tensor
        output = np.dot(input_tensor, self.weights)
        return output.copy()
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value
        
    def backward(self, error_tensor):
        # Compute gradient with respect to weights
        self._gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        
        # Compute gradient with respect to input tensor (excluding bias term)
        gradient_input = np.dot(error_tensor, self.weights.T)
        return gradient_input[:, :-1]  # Exclude the bias term
    
    