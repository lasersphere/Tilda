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
        f.preProc('../AnaDB.sqlite')
        self.assertEqual(f.nrScalers, 1)
        self.assertEqual(f.nrTracks, 1)
        
    def test_x(self):
        f = KepcoImporterTLD('../testKepco.txt')
        f.preProc('../AnaDB.sqlite')
        np.testing.assert_array_equal(f.x, [[2, 3, 5, 7, 8]])
        np.testing.assert_array_equal(f.getSingleSpec(0, -1)[0], [2, 3, 5, 7, 8])

    def test_y(self):
        f = KepcoImporterTLD('../testKepco.txt')
        f.preProc('../AnaDB.sqlite')
        np.testing.assert_array_equal(f.cts, [[[400, 900, 2500, 4900, 6400]]])
        np.testing.assert_array_equal(f.getSingleSpec(0, -1)[1], [400, 900, 2500, 4900, 6400])
        
    def test_err(self):
        f = KepcoImporterTLD('../testKepco.txt')
        f.preProc('../AnaDB.sqlite')
        np.testing.assert_allclose(f.err, [[[0.04,  0.09,  0.25,  0.49,  0.64]]])
        np.testing.assert_allclose(f.getSingleSpec(0, -1)[2], [0.04,  0.09,  0.25,  0.49,  0.64])

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()