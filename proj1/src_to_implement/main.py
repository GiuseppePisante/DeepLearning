from pattern import Circle
import numpy as np  
import matplotlib.pyplot as plt
import json
from generator import ImageGenerator
import unittest
import tabulate
import argparse

class TestImageGenerator(unittest.TestCase):
    def test_unique_samples(self):
        gen = ImageGenerator('./exercise_data/', './Labels.json', 50, [32, 32, 3], rotation=False, mirroring=False, shuffle=False)
        b1 = gen.next()
        b2 = gen.next()
        sample_index = np.random.choice(np.arange(50))
        sample = b1[0][sample_index]
        b1_without_sample = np.delete(b1[0], sample_index, axis=0)
        
        self.assertFalse(np.any(np.all(sample == b1_without_sample, axis=(1, 2, 3))),
                         msg="Possible error: One or more samples appear more than once in the first batch (even for non-overlapping batches). Please make sure that all samples are unique within the batch.")
        
        self.assertFalse(np.any(np.all(sample == b2[0], axis=(1, 2, 3))),
                         msg="Possible error: One or more samples appear more than once in the following batches (even for non-overlapping batches). Please make sure that all samples are unique within the batch.")

if __name__ == "__main__":
    unittest.main()