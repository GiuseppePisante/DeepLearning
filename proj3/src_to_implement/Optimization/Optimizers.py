import numpy as np

class Sgd():
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        weight_tensor = weight_tensor - self.learning_rate * gradient_tensor
        return weight_tensor


class SgdWithMomentum():
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.mu = 0.9 # other options: 0.95, 0.99
        self.velocity = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.velocity is None:
            self.velocity = np.zeros_like(weight_tensor)
        self.velocity = self.mu * self.velocity - self.learning_rate * gradient_tensor 
        weight_tensor = weight_tensor + self.velocity
        return weight_tensor


class Adam():
    def __init__(self, learning_rate = 0.001, mu = 0.9, rho = 0.999):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.velocity = None
        self.squared_gradient_avg = None
        self.iteration = 0 # to do bias correction

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.velocity is None:
            self.velocity = np.zeros_like(weight_tensor)
        if self.squared_gradient_avg is None:
            self.squared_gradient_avg = np.zeros_like(gradient_tensor)
        self.iteration += 1
        self.velocity = self.mu * self.velocity + (1 - self.mu) * gradient_tensor   
        self.squared_gradient_avg = self.rho * self.squared_gradient_avg + (1 - self.rho) * gradient_tensor **2   
        velocity_corrected = self.velocity / (1 - self.mu ** self.iteration)
        squared_gradient_avg_corrected = self.squared_gradient_avg / (1 - self.rho ** self.iteration)
        epsilon = 1e-8
        weight_tensor = weight_tensor - self.learning_rate * velocity_corrected / (np.sqrt(squared_gradient_avg_corrected) + epsilon)
        return weight_tensor



