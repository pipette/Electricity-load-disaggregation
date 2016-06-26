'''
Unit tests for the Factorial HMM module
To run the tests: make test
'''
from FHMM import combine_means, combine_parameters
from Clustering import cluster
import numpy as np
import unittest as unittest
import numpy as np


class TestFHMM(unittest.TestCase):

    def test_combine_means(self):
        function_outcome = combine_means([[1,2],[3,4]])[0]
        expected_outcome = np.array([[4],[5],[5],[6]])
        return self.assertTrue(np.array_equal(function_outcome, expected_outcome))

    def test_cluster(self):
        function_outcome = cluster(np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3]).reshape((-1,1)),4)
        expected_outcome = np.array([[ 1.],[ 2.],[ 3.]])
        return self.assertTrue(np.array_equal(np.sort(function_outcome, axis = 0), expected_outcome))

if __name__ == '__main__':
    unittest.main()


