'''
Created on 23.04.2014

@author: hammen
'''
import unittest

from Spectra.Voigt import Voigt

class Test_Voigt(unittest.TestCase):

    def setUp(self):
        class Expando(object): pass
        self.iso = Expando()
        self.iso.shape = {'gau': 50, 'lor': 10}
        self.iso.fixShape = {'gau': False, 'lor': False}
        
        self.line = Voigt(self.iso)

    def test_evaluateSide(self):
        self.assertAlmostEqual(self.line.evaluate([300], [50, 10]), 0.0056614479422570484, 6)

    def test_evaluateFlank(self):
        self.assertAlmostEqual(self.line.evaluate([70], [50, 10]), 0.4423914063149442, 6)
        
    def test_evaluateCentre(self):
        self.assertAlmostEqual(self.line.evaluate([0], [50, 10]), 1, 6)
        
    def test_getPars(self):
        self.assertEqual(self.line.getPars(), [50, 10])
        
    def test_nPar(self):
        self.assertEqual(self.line.nPar, len(self.line.getPars()))
        
    def test_edges(self):
        self.assertEqual(self.line.leftEdge(), -300)
        self.assertEqual(self.line.rightEdge(), 300)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()