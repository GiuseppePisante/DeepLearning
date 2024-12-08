import numpy as np
from Optimization.Optimizers import Optimizer

class L2_Regularizer(Optimizer):
    def __init__(self, alpha):
        self.alpha = alpha
        super().__init__()

    def calculate_gradient(self, weight):
        return self.alpha * weight

    def norm(self, weights):
        return self.alpha * np.sum(weights**2)

class L1_Regularizer(Optimizer):
    def __init__(self, alpha):
        self.alpha = alpha
        super().__init__()

    def calculate_gradient(self, weight):
        return self.alpha * np.sign(weight)

    def norm(self, weights):
        return self.alpha * np.sum(np.abs(weights))
