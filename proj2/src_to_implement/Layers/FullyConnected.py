from Base import BaseLayer
import numpy as np

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        BaseLayer.__init__()
        self.trainable = True
        self.weights = np.random.rand(input_size, output_size)
        
    def forward(self, input_tensor):
        output = self.weights @ input_tensor
        return output.copy() # ! non necessario ?
    
    # TODO: guardare le priority, setter e getter