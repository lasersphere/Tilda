'''
Created on 28.04.2014

@author: hammen
'''
import unittest

from DBIsotope import DBIsotope

class Test(unittest.TestCase):
    def test_frequency(self):
        iso = DBIsotope("1_Mi-D0", "iso.sqlite")
        self.assertEqual(iso.freq, 1000)
        
    def test_J(self):
        iso = DBIsotope("1_Mi-D0", "iso.sqlite")
        self.assertEqual(iso.Jl, 0.5)
        self.assertEqual(iso.Ju, 1.5)
        
    def test_fix(self):
        iso = DBIsotope("1_Mi-D0", "iso.sqlite")
        self.assertEqual(iso.fixArat, False)
        
    def test_Br(self):
        iso = DBIsotope("1_Mi-D0", "iso.sqlite")
        self.assertEqual(iso.Bu, 30)
        
    def test_doubleIso(self):
        iso = DBIsotope("2_Mi-D0", "iso.sqlite")
        self.assertEqual(iso.mass, 2)
        self.assertEqual(iso.m.mass, 3)
        
    def test_relInt(self):
        iso = DBIsotope("3_Mi-D0", "iso.sqlite")
        self.assertEqual(iso.relInt, [1, 0.5])


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()