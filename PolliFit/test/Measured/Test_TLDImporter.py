'''
Created on 05.05.2014

@author: hammen
'''
import unittest

import numpy as np

from Measurement.TLDImporter import TLDImporter

class Test_TLDImporter(unittest.TestCase):


    def test_import(self):
        f = TLDImporter('../Project/Data/testTLD.tld')
        f.preProc('../Project/AnaDB.sqlite')
        self.assertEqual(f.nrScalers, 2)
        self.assertEqual(f.nrTracks, 1)
        self.assertEqual(f.col, False)

    def test_x(self):
        f = TLDImporter('../Project/Data/testTLD.tld')
        f.preProc('../Project/AnaDB.sqlite')
        # np.testing.assert_array_equal(f.x, [[20-(2.0/50*2+0.5+10),20-(3.0/50*2+0.5+10), 20-(5.0/50*2+0.5+10), 20-(7.0/50*2+0.5+10), 20-(8.0/50*2+0.5+10)]])
        # np.testing.assert_array_equal(f.getSingleSpec(0, -1)[0],[20-(2.0/50*2+0.5+10),20-(3.0/50*2+0.5+10), 20-(5.0/50*2+0.5+10), 20-(7.0/50*2+0.5+10), 20-(8.0/50*2+0.5+10)])

    def test_y(self):
        f = TLDImporter('../Project/Data/testTLD.tld')
        f.preProc('../Project/AnaDB.sqlite')
        np.testing.assert_array_equal(f.cts, [[[4, 9, 25, 49, 64],[81,100,121,144,169]]])
        np.testing.assert_array_equal(f.getSingleSpec(0, -1)[1], [4, 9, 25, 49, 64])

    def test_err(self):
        f = TLDImporter('../Project/Data/testTLD.tld')
        f.preProc('../Project/AnaDB.sqlite')
        np.testing.assert_array_equal(f.err, [[[2, 3, 5, 7, 8],[9,10,11,12,13]]])
        np.testing.assert_array_equal(f.getSingleSpec(0, -1)[2], [2, 3, 5, 7, 8])

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()