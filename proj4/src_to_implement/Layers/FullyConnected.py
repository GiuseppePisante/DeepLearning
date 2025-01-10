from .Base import BaseLayer
import numpy as np
from copy import *


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True
        self._optimizer = None
        self.weights = np.random.rand(input_size + 1, output_size)
        self._gradient_weights = None
        
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

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value
        
    def backward(self, error_tensor):
        ## Change
        self.error_tensor = error_tensor
        # Compute gradient with respect to weights
        self._gradient_weights = np.dot(self.input_tensor.T, error_tensor)

        # Update weights if optimizer is set
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        
        # Compute gradient with respect to input tensor (excluding bias term)
        gradient_input = np.dot(error_tensor, self.weights.T)
        
        # Remove bias term from gradient_input
        gradient_input = gradient_input[:, :-1]
        
        
        return gradient_input
    

    def initialize(self, weights_initializer, bias_initializer):
        weights = weights_initializer.initialize(np.shape(self.weights[:-1, :]), np.shape(self.weights[:-1, :])[0],
                                                 np.shape(self.weights[:-1, :])[1])
        bias = np.expand_dims(self.weights[-1, :], axis=0)
        bias = bias_initializer.initialize(bias.shape, bias.shape[0], bias.shape[1])
        self.weights = np.concatenate((weights, bias), axis=0)
        return self.weights, bias

    def set_optimizer(self, optimizer):
        self.optimizer = deepcopy(optimizer)
        #return self.optimizer


