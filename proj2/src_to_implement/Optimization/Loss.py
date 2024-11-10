import numpy as np

class CrossEntropyLoss():
    def __init__(self):
        self.loss = None
        pass

    def forward(self, prediction_tensor, label_tensor):
        self.loss = np.sum(-np.log(prediction_tensor[label_tensor == 1] + np.finfo(float).eps))