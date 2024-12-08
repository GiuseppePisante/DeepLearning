import numpy as np

class Constant():
    def __init__(self, const_value = 0.1):
        self.const_value = const_value
        self.weights = None
        

    def initialize(self, weights_shape, fan_in, fan_out):
        self.weights = np.ones(weights_shape) * self.const_value 
        return self.weights.copy()
    



class UniformRandom():
    def __init__(self):
        self.weights = None
        

    def initialize(self, weights_shape, fan_in, fan_out):
        self.weights = np.random.uniform(0, 1, size=weights_shape)
        return self.weights.copy()
    


class Xavier():
    def __init__(self):
        self.weights = None
        self.sigma = None
        

    def initialize(self, weights_shape, fan_in, fan_out):
        self.sigma = np.sqrt(2 / (fan_in + fan_out))
        self.weights = np.random.randn(*weights_shape) * self.sigma
        return self.weights.copy()
    


class He():
    def __init__(self):
        self.weights = None
        self.sigma = None
        

    def initialize(self, weights_shape, fan_in, fan_out):
        self.sigma = np.sqrt(2 / fan_in)
        self.weights = np.random.randn(*weights_shape) * self.sigma
        return self.weights.copy()
    
