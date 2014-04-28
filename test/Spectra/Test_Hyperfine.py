'''
Created on 23.04.2014

@author: hammen
'''
import unittest

import numpy as np

from Spectra.Voigt import Voigt
from Spectra.Hyperfine import Hyperfine

class Test_Hyperfine(unittest.TestCase):

    def setUp(self):
        class Expando(object): pass
        self.iso = Expando()
        self.iso.shape = {'gau': 50, 'lor': 10}
        self.iso.fixShape = {'gau': False, 'lor': False}
        self.iso.intScale = 1
        self.iso.shift = 99
        self.iso.Al = 100
        self.iso.Bl = 101
        self.iso.Ar = 102
        self.iso.Br = 103

        self.iso.I = 0
        self.iso.Jl = 0.5
        self.iso.Ju = 1.5
        
        self.shape = Voigt(self.iso)
        self.line = Hyperfine(self.iso, self.shape)

    def test_getPars(self):
        np.testing.assert_almost_equal(self.line.getPars(), [99, 100, 101, 102, 103, 1], 3)
        
    def test_nPar(self):
        self.assertEqual(self.line.nPar, len(self.line.getPars()))

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()