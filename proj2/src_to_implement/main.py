import unittest
import numpy as np
from Optimization import Optimizers
import NeuralNetworkTests


        
def main():
    #prova = TestOptimizers1()
    #prova.test_sgd()

    A = np.ones([3,2])
    B = np.zeros([3,3])
    B[1,1] = 2
    A[1,1] = 4
    pos = [B == 2]
    print("elementto: ", A[B == 2])
    


# Run the main function
if __name__ == "__main__":
    main()
