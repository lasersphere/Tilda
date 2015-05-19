'''
Created on 23.04.2014

@author: hammen
'''
import unittest

import Physics

class Test_Physics(unittest.TestCase):


    def test_relVelocity(self):
        self.assertAlmostEqual(Physics.relVelocity(1e-13, 1.5e-27), 1.15405860588344e07, 1)

    def test_relDoppler(self):
        pass
        
    def test_hypCoeffZero(self):
        self.assertEqual(Physics.HFCoeff(0, 0.5, 0.5), (0, 0))
        
    def test_hypCoeffNonZero(self):
        self.assertEqual(Physics.HFCoeff(1, 1.5, 2.5), (1.5, 0.25))

    def test_calcHFTransZeroI(self):
        self.assertEqual(Physics.HFTrans(0, 0.5, 1.5), [(0.5, 1.5, 0.0, 0.0, 0.0, 0.0)])
    
    def test_calcHFTransNonzeroI(self):
        self.assertEqual(Physics.HFTrans(0.5, 0.5, 1.5), [(0., 1., -0.75, 0.0, -1.25, 0.0), (1., 1., 0.25, 0.0, -1.25, 0.0), (1., 2., 0.25, 0.0, 0.75, 0.0)])
        
    def test_calcHFLineSplit(self):
        self.assertEqual(Physics.HFLineSplit(1, 2, 3, 4, [(0.5, 1.5, 0.0, 0.0, 0.0, 0.0)]), [0.])
        
    def test_calcHFLineSplit2(self):
        self.assertEqual(Physics.HFLineSplit(1, 2, 3, 4, [(0., 1., -2.25, 0.0, 0.25, 0.0)]), [3.])
    
    
    
    

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()