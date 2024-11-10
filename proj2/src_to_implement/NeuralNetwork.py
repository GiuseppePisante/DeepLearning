import numpy as np

class NeuralNetwork():
    def __init__(self, input_optimizer):
        self.optimizer = input_optimizer
        self.loss = []
        self.layer = []
        self.data_layer = None
        self.loss_layer = None

    def forward(self):
        self.layer.forward(self.data_layer)
    
    def backward(self, label_tensor):
        self.error_tensor = self.loss_layer.backward(label_tensor)

    def append_layer(self, layer):
        if layer.trainable():
            self.deep_copy = 

    
