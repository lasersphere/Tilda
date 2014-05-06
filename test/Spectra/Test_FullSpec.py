'''
Created on 23.04.2014

@author: hammen
'''
import unittest

import numpy as np

from Spectra.FullSpec import FullSpec
from DBIsotope import DBIsotope

class Test_FullSpec(unittest.TestCase):

    def setUp(self):
        self.iso = DBIsotope('1_Mi-D0', '../iso.sqlite')
        self.line = FullSpec(self.iso)

    def test_getPars(self):
        np.testing.assert_almost_equal(self.line.getPars(), [0, 50, 10, 3, 101, 20, 102, 30, 999], 3)
        
    def test_getParNames(self):
        self.assertEqual(self.line.getParNames(), ['offset', 'sigma', 'gamma', 'Gcenter', 'GAl', 'GBl', 'GAu', 'GBu', 'GInt0'])

    def test_getFixed(self):
        self.assertEqual(self.line.getFixed(), [False, True, True, False, True, True, True, True, False])
        
    def test_nPar(self):
        self.assertEqual(self.line.nPar, len(self.line.getPars()))

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()