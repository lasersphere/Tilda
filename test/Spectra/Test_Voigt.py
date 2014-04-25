'''
Created on 23.04.2014

@author: hammen
'''
import unittest

from Spectra.Voigt import Voigt

class Test(unittest.TestCase):

    def setUp(self):
        class Expando(object): pass 
        self.iso = Expando()
        self.iso.gauWidth = 50
        self.iso.fixGauss = True
        self.iso.lorWidth = 10
        self.iso.fixLor = True
        self.line = Voigt(self.iso)

    def test_evaluateSide(self):
        self.assertAlmostEqual(self.line.evaluate([300], [50, 10]), 3.877908752409212e-5, 7)

    def test_evaluateFlank(self):
        self.assertAlmostEqual(self.line.evaluate([70], [50, 10]), 3.030238066368942e-3, 7)
        
    def test_evaluateCentre(self):
        self.assertAlmostEqual(self.line.evaluate([0], [50, 10]), 6.849676605633871e-3, 7)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()