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
        self.assertTrue(True)
        
    def test_hypCoeff(self):
        pass
    
    def test_calcHFTransNonzeroI(self):
        pass
    
    def test_calcHFTransZeroI(self):
        pass
    
    def test_calcHFLinePos(self):
        pass
    
    
    
    

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()