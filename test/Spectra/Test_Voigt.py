'''
Created on 23.04.2014

@author: hammen
'''
import unittest

from Spectra.Voigt import Voigt

class Test(unittest.TestCase):

    def setUp(self):
        self.line = Voigt()

    def test_evaluateSide(self):
        self.assertAlmostEqual(self.line.evaluate([300], [50, 10]), 3.877908752409212e-5, 7)

    def test_evaluateFlank(self):
        self.assertAlmostEqual(self.line.evaluate([70], [50, 10]), 3.030238066368942e-3, 7)
        
    def test_evaluateCentre(self):
        self.assertAlmostEqual(self.line.evaluate([0], [50, 10]), 6.849676605633871e-3, 7)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()