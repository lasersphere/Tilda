'''
Created on 23.04.2014

@author: hammen
'''
import unittest

import numpy as np

from Spectra.Voigt import Voigt
from Spectra.Hyperfine import Hyperfine
from DBIsotope import DBIsotope

class Test_Hyperfine(unittest.TestCase):

    def setUp(self):

        self.iso = DBIsotope("1_Mi-D0", "../iso.sqlite")
        self.iso2 = DBIsotope("3_Mi-D0", "../iso.sqlite")

        self.shape = Voigt(self.iso)
        

    def test_getPars(self):
        line = Hyperfine(self.iso, self.shape)
        np.testing.assert_almost_equal(line.getPars(), [3, 101, 20, 102, 30, 999], 3)
        
    def test_nPar(self):
        line = Hyperfine(self.iso, self.shape)
        self.assertEqual(line.nPar, len(line.getPars()))
        
    def test_leftEdge(self):
        line = Hyperfine(self.iso, self.shape)
        self.assertEqual(line.leftEdge(), -300)
        self.assertEqual(line.rightEdge(), 300)
        
    def test_NonZeroI(self):
        pass

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()