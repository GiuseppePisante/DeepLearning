import numpy as np
class CrossEntropyLoss():
    def __init__(self):
        self.loss = None
        self.prediction_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        
        # Compute the loss with added epsilon for stability
        self.loss = np.sum(-np.log(prediction_tensor[label_tensor == 1] + np.finfo(float).eps))
        return self.loss
        
    def backward(self, label_tensor):
        # Avoid division by zero in backward pass
        return -label_tensor / (self.prediction_tensor + np.finfo(float).eps)
