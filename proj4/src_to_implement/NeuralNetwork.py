import numpy as np
import copy
from sklearn.preprocessing import OneHotEncoder

class NeuralNetwork():
    def __init__(self, input_optimizer, weights_initializer, bias_initializer, ):
        self.optimizer = input_optimizer
        self.loss = []
        self.layers = []
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.data_layer = None
        self.loss_layer = None
        self.label_tensor = None
        self._phase = None

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        self._phase = value
        for layer in self.layers:
            layer.phase = value

    def forward(self):
        input_tensor, self.label_tensor = self.data_layer.next()
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        regularization_loss = 0
        if self.optimizer.regularizer is not None:
            for layer in self.layers:
                if layer.trainable:
                    regularization_loss += self.optimizer.regularizer.norm(layer.weights)
        return self.loss_layer.forward(input_tensor, self.label_tensor) + regularization_loss
    
    def backward(self, label_tensor):
        error_tensor = self.loss_layer.backward(label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            layer.initialize(copy.deepcopy(self.weights_initializer), copy.deepcopy(self.bias_initializer))
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        self.phase = True  # Training phase
        for _ in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward(self.label_tensor)

    def test(self, input_tensor):
        self.phase = False  # Testing phase
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor





