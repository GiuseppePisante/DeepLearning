import numpy as np

class CrossEntropyLoss():
    def __init__(self):
        self.loss = None
        self.prediction_tensor = None  # Store prediction tensor
        pass

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor  # Store prediction tensor
        self.loss = np.sum(-np.log(prediction_tensor[label_tensor == 1] + np.finfo(float).eps))
        return self.loss  # Return the loss value
        
    def backward(self, label_tensor):
        return -label_tensor / (self.prediction_tensor + np.finfo(float).eps)  # Use stored prediction tensor