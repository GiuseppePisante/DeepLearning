import numpy as np
import copy
from sklearn.preprocessing import OneHotEncoder

class NeuralNetwork():
    def __init__(self, input_optimizer):
        self.optimizer = input_optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.label_tensor = None

    def forward(self):
        input_tensor, self.label_tensor = self.data_layer.next()
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return self.loss_layer.forward(input_tensor, self.label_tensor)
    
    def backward(self, label_tensor):
        error_tensor = self.loss_layer.backward(label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        for _ in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward(self.label_tensor)

    def test(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor  # Return the final output tensor directly

    def set_data_layer(self, data_layer):
        self.data_layer = data_layer
        # Ensure the label tensor is correctly initialized
        self.data_layer._label_tensor = OneHotEncoder(sparse_output=False).fit_transform(
            self.data_layer._data.target.reshape(-1, 1)
        )


