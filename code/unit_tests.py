'''
Unit tests for the Factorial HMM module
To run the tests: make test
'''
from FHMM import combine_means, combine_parameters
import numpy as np
import unittest as unittest
import numpy as np


class TestFHMM(unittest.TestCase):

    def test_combine_means(self):
        function_outcome = combine_means([[1,2],[3,4]])
        expected_outcome = np.array([[4],[5],[5],[6]])
        return self.assertTrue(np.array_equal(function_outcome, expected_outcome))


if __name__ == '__main__':
    unittest.main()


