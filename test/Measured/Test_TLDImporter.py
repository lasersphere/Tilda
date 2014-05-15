'''
Created on 05.05.2014

@author: hammen
'''
import unittest
import Experiment

import numpy as np

from Measurement.TLDImporter import TLDImporter

class Test_TLDImporter(unittest.TestCase):


    def test_import(self):
        f = TLDImporter('../testTLD.txt')
        self.assertEqual(f.nrScalers, 2)
        self.assertEqual(f.nrTracks, 1)
        
    def test_x(self):
        f = TLDImporter('../testTLD.txt')
        np.testing.assert_array_equal(f.x, [[Experiment.getAccVolt()-(Experiment.lineToScan(2/50)-998.92), Experiment.getAccVolt()-(Experiment.lineToScan(3/50)-998.92), Experiment.getAccVolt()-(Experiment.lineToScan(5/50)-998.92), Experiment.getAccVolt()-(Experiment.lineToScan(7/50)-998.92), Experiment.getAccVolt()-(Experiment.lineToScan(8/50)-998.92)]])
        np.testing.assert_array_equal(f.getSingleSpec(0, -1)[0],[Experiment.getAccVolt()-(Experiment.lineToScan(2/50)-998.92), Experiment.getAccVolt()-(Experiment.lineToScan(3/50)-998.92), Experiment.getAccVolt()-(Experiment.lineToScan(5/50)-998.92), Experiment.getAccVolt()-(Experiment.lineToScan(7/50)-998.92), Experiment.getAccVolt()-(Experiment.lineToScan(8/50)-998.92)])

    def test_y(self):
        f = TLDImporter('../testTLD.txt')
        np.testing.assert_array_equal(f.cts, [[[4, 9, 25, 49, 64]],[[81,100,121,144,169]]])
        np.testing.assert_array_equal(f.getSingleSpec(0, -1)[1], [4, 9, 25, 49, 64])
        
    def test_err(self):
        f = TLDImporter('../testTLD.txt')
        np.testing.assert_array_equal(f.err, [[[2, 3, 5, 7, 8]],[[9,10,11,12,13]]])
        np.testing.assert_array_equal(f.getSingleSpec(0, -1)[2], [2, 3, 5, 7, 8])

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()