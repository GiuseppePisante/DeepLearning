import unittest
import numpy as np
from Optimization import Optimizers
import NeuralNetworkTests

class TestOptimizers1(unittest.TestCase):

    def test_sgd(self):
        optimizer = Optimizers.Sgd(1.)

        result = optimizer.calculate_update(1., 1.)
        np.testing.assert_almost_equal(result, np.array([0.]),
                                       err_msg="Possible error: The Sgd optimizer is not properly implemented. "
                                               "SGD is used by some other unittests. If these fail it could be caused "
                                               "by a wrong implementation of the SGD optimizer.")

        result = optimizer.calculate_update(result, 1.)
        np.testing.assert_almost_equal(result, np.array([-1.]),
                                       err_msg="Possible error: The Sgd optimizer is not properly implemented. "
                                               "SGD is used by some other unittests. If these fail it could be caused "
                                               "by a wrong implementation of the SGD optimizer."
                                       )
        
def main():
    #prova = TestOptimizers1()
    #prova.test_sgd()

    A = np.ones(3,3)
    B = np.zeros(3,3)
    


# Run the main function
if __name__ == "__main__":
    main()
