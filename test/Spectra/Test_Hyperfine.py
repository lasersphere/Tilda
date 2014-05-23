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

        self.iso = DBIsotope('40_Ca', 'Ca-D1', "../Project/AnaDB.sqlite")
        self.iso2 = DBIsotope('42_Ca', 'Ca-D1', "../Project/AnaDB.sqlite")

        self.shape = Voigt(self.iso)
        

    def test_getPars(self):
        line = Hyperfine(self.iso, self.shape)
        np.testing.assert_almost_equal(line.getPars(), [1., 2., 3., 4., 5., 1000.], 3)
        
    def test_nPar(self):
        line = Hyperfine(self.iso, self.shape)
        self.assertEqual(line.nPar, len(line.getPars()))
        
    def test_leftEdge(self):
        line = Hyperfine(self.iso, self.shape)
        self.assertEqual(line.leftEdge(), -250)
        self.assertEqual(line.rightEdge(), 250)
        
    def test_NonZeroI(self):
        pass

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()