'''
Created on 05.05.2014

@author: hammen
'''
import unittest

import numpy as np

from Measurement.SimpleImporter import SimpleImporter

class Test_SimpleImporter(unittest.TestCase):


    def test_import(self):
        f = SimpleImporter('../test.txt', 10, 12586, True)
        self.assertEqual(f.nrScalers, 1)
        self.assertEqual(f.nrTracks, 1)
        
    def test_x(self):
        f = SimpleImporter('../test.txt', 10, 12586, True)
        np.testing.assert_array_equal(f.x, [[8, 7, 5, 3, 2]])
        np.testing.assert_array_equal(f.getSingleSpec(0, -1)[0], [8, 7, 5, 3, 2])

    def test_y(self):
        f = SimpleImporter('../test.txt', 10, 12586, True)
        np.testing.assert_array_equal(f.cts, [[[4, 9, 25, 49, 64]]])
        np.testing.assert_array_equal(f.getSingleSpec(0, -1)[1], [4, 9, 25, 49, 64])
        
    def test_err(self):
        f = SimpleImporter('../test.txt',10, 12586, True)
        np.testing.assert_array_equal(f.err, [[[2, 3, 5, 7, 8]]])
        np.testing.assert_array_equal(f.getSingleSpec(0, -1)[2], [2, 3, 5, 7, 8])

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()