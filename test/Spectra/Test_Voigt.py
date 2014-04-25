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
        self.assertAlmostEqual(self.line.evaluate([300], [50, 10]), 0.0056614479422570484, 6)

    def test_evaluateFlank(self):
        self.assertAlmostEqual(self.line.evaluate([70], [50, 10]), 0.4423914063149442, 6)
        
    def test_evaluateCentre(self):
        self.assertAlmostEqual(self.line.evaluate([0], [50, 10]), 1, 6)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()