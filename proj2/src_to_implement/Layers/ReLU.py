from .Base import BaseLayer
import numpy as np

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        # Add bias term to input tensor
        bias = np.ones((input_tensor.shape[0], 1))
        input_tensor = np.hstack((input_tensor, bias))
        self.input_tensor = input_tensor
        output = np.dot(input_tensor, self.weights)
        output = np.maximum(0, output)
        return output.copy()
    
    def backward(self, error_tensor):
        # Compute gradient with respect to weights
        self._gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        
        # Compute gradient with respect to input tensor (excluding bias term)
        gradient_input = np.dot(error_tensor, self.weights.T)
        
        # Remove bias term from gradient_input
        gradient_input = gradient_input[:, :-1]
        
        # Update weights if optimizer is set
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        
        return gradient_input
    
    

