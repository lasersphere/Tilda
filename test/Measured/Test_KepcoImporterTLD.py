'''
Created on 05.05.2014

@author: hammen
'''
import unittest

import numpy as np

from Measurement.KepcoImporterTLD import KepcoImporterTLD

class Test_KepcoImporterTLD(unittest.TestCase):


    def test_import(self):
        f = KepcoImporterTLD('../testKepco.txt')
        self.assertEqual(f.nrScalers, 1)
        self.assertEqual(f.nrTracks, 1)
        
    def test_x(self):
        f = KepcoImporterTLD('../testKepco.txt')
        np.testing.assert_array_equal(f.x, [[2, 3, 5, 7, 8]])
        np.testing.assert_array_equal(f.getSingleSpec(0, -1)[0], [2, 3, 5, 7, 8])

    def test_y(self):
        f = KepcoImporterTLD('../testKepco.txt')
        np.testing.assert_array_equal(f.cts, [[[4, 9, 25, 49, 64]]])
        np.testing.assert_array_equal(f.getSingleSpec(0, -1)[1], [4, 9, 25, 49, 64])
        
    def test_err(self):
        f = KepcoImporterTLD('../testKepco.txt')
        np.testing.assert_array_equal(f.err, [[[10**-4,10**-4,10**-4,10**-4,10**-4]]])
        np.testing.assert_array_equal(f.getSingleSpec(0, -1)[2], [10**-4,10**-4,10**-4,10**-4,10**-4])

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()