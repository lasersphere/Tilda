"""
Created on 

@author: simkaufm

Module Description: Module for testing the XMLImporter
"""

import unittest

import numpy as np

from Measurement.XMLImporter import XMLImporter


class Test_XMLImporter(unittest.TestCase):

    def test_import(self):
        f = XMLImporter('../Project/Data/testTilda.xml')
        f.preProc('../Project/tildaDB.sqlite')
        self.assertEqual(f.nrScalers, [3])
        self.assertEqual(f.nrTracks, 1)

    def test_x(self):
        f = XMLImporter('../Project/Data/testTilda.xml')
        f.preProc('../Project/tildaDB.sqlite')
        x = np.arange(0, 16394 * 7, 16393)
        np.testing.assert_array_equal(f.x, [x])
        np.testing.assert_array_equal(f.getSingleSpec(0, -1)[0], x)

    def test_y(self):
        f = XMLImporter('../Project/Data/testTilda.xml')
        f.preProc('../Project/tildaDB.sqlite')
        y = [np.arange(1, 9, 1), np.arange(2, 10, 1), np.arange(3, 11, 1)]
        np.testing.assert_array_equal(f.cts, [y])
        np.testing.assert_array_equal(f.getSingleSpec(0, -1)[1], y[0])

    def test_err(self):
        f = XMLImporter('../Project/Data/testTilda.xml')
        f.preProc('../Project/tildaDB.sqlite')
        err = np.sqrt([np.arange(1, 9, 1), np.arange(2, 10, 1), np.arange(3, 11, 1)])
        np.testing.assert_array_equal(f.err, [err])
        np.testing.assert_array_equal(f.getSingleSpec(0, -1)[2], err[0])

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()