"""
Created on 

@author: simkaufm

Module Description: Module for testing the XMLImporter
"""

import unittest
from copy import deepcopy
import numpy as np
from Measurement.XMLImporter import XMLImporter
import MPLPlotter as mplplot
from SPFitter import SPFitter
from Spectra.Straight import Straight


class Test_XMLImporter(unittest.TestCase):
    def test_import_cs(self):
        f = XMLImporter('../Project/Data/testTildaCs.xml')
        f.preProc('../Project/tildaDB109.sqlite')
        self.assertEqual(f.nrScalers, [3])
        self.assertEqual(f.nrTracks, 1)
        self.assertEqual(f.getNrSteps(0), 21)

    def test_import_trs(self):
        f = XMLImporter('../Project/Data/testTildaTrs.xml')
        f.preProc('../Project/tildaDB109.sqlite')
        self.assertEqual(f.nrScalers, [4])
        self.assertEqual(f.nrTracks, 1)
        self.assertEqual(f.getNrSteps(0), 5)

    def test_x_cs(self):
        f = XMLImporter('../Project/Data/testTildaCs.xml')
        f.preProc('../Project/tildaDB109.sqlite')
        x = [np.array([30499.66005, 30449.6125, 30399.56495, 30349.5174,
                       30299.46985, 30249.4223, 30199.37475, 30149.3272,
                       30099.27965, 30049.2321, 29999.18455, 29949.137,
                       29899.08945, 29849.0419, 29798.99435, 29748.9468,
                       29698.89925, 29648.8517, 29598.80415, 29548.7566,
                       29498.70905])]
        np.testing.assert_allclose(f.x, x, rtol=1e-9)
        np.testing.assert_allclose(f.getSingleSpec(0, -1)[0], x[0], rtol=1e-9)
        np.testing.assert_allclose(f.getArithSpec([0], -1)[0], x[0], rtol=1e-9)  # New not tested yet...

    def test_x_trs(self):
        f = XMLImporter('../Project/Data/testTildaTrs.xml')
        f.preProc('../Project/tildaDB109.sqlite')
        x = [[30499.66005, 30374.6782, 30249.69635, 30124.7145, 29999.73265]]
        np.testing.assert_allclose(f.x, x, rtol=1e-9)
        np.testing.assert_allclose(f.getSingleSpec(0, -1)[0], x[0], rtol=1e-9)
        np.testing.assert_allclose(f.getArithSpec([0], -1)[0], x[0], rtol=1e-9)  # New not tested yet...

    def test_y_cs(self):
        f = XMLImporter('../Project/Data/testTildaCs.xml')
        f.preProc('../Project/tildaDB109.sqlite')
        y = [[np.arange(1, 22, 1), np.arange(2, 23, 1), np.arange(3, 24, 1)]]
        np.testing.assert_array_equal(f.cts, y)
        np.testing.assert_array_equal(f.getSingleSpec(0, 0)[1], y[0][0])
        np.testing.assert_array_equal(f.getArithSpec([0], 0)[1], y[0][0])    # New not tested yet...

    def test_y_trs(self):
        f = XMLImporter('../Project/Data/testTildaTrs.xml')
        f.preProc('../Project/tildaDB109.sqlite')
        y = [np.full((4, 5), 250)]
        np.testing.assert_array_equal(f.cts, y)
        np.testing.assert_array_equal(f.getSingleSpec(0, 0)[1], y[0][0])
        np.testing.assert_array_equal(f.getArithSpec([0], 0)[1], y[0][0])    # New not tested yet...

    def test_t_trs(self):
        f = XMLImporter('../Project/Data/testTildaTrs.xml')
        f.preProc('../Project/tildaDB109.sqlite')
        t_tes = [np.arange(0, 25000, 10)]
        np.testing.assert_equal(f.t, t_tes)
        np.testing.assert_equal(f.get_scaler_step_and_bin_num(0), (4, 5, 2500))

    def test_time_res_trs(self):
        f = XMLImporter('../Project/Data/testTildaTrs.xml')
        f.preProc('../Project/tildaDB109.sqlite')
        time_res_1pmt_1step_1tr = np.zeros(2500)
        mask = np.arange(0, 2500, 10)
        np.put(time_res_1pmt_1step_1tr, mask, np.full(mask.shape, 1))
        time_res_tes_full = [np.zeros((4, 5, 2500))]
        for pmt_ind, pmt_arr in enumerate(time_res_tes_full[0]):
            for step_ind, step_arr in enumerate(pmt_arr):
                time_res_tes_full[0][pmt_ind][step_ind] = deepcopy(time_res_1pmt_1step_1tr)
        np.testing.assert_equal(f.time_res, time_res_tes_full)

    def test_t_proj_trs(self):
        f = XMLImporter('../Project/Data/testTildaTrs.xml')
        f.preProc('../Project/tildaDB109.sqlite')
        time_proj_1pmt_1step_1tr = np.zeros(2500)
        mask = np.arange(0, 2500, 10)
        np.put(time_proj_1pmt_1step_1tr, mask, np.full(mask.shape, 5))
        t_proj_full = [np.zeros((4, 2500))]
        for pmt_ind, pmt_arr in enumerate(t_proj_full[0]):
            t_proj_full[0][pmt_ind] = deepcopy(time_proj_1pmt_1step_1tr)
        np.testing.assert_equal(f.t_proj, t_proj_full)

    def test_err_cs(self):
        f = XMLImporter('../Project/Data/testTildaCs.xml')
        f.preProc('../Project/tildaDB109.sqlite')
        err = np.sqrt([[np.arange(1, 22, 1), np.arange(2, 23, 1), np.arange(3, 24, 1)]])
        np.testing.assert_array_equal(f.err, err)
        np.testing.assert_array_equal(f.getSingleSpec(0, -1)[2], err[0][0])
        np.testing.assert_array_equal(f.getArithSpec([0], -1)[2], err[0][0]) # New not tested yet...

    def test_err_trs(self):
        f = XMLImporter('../Project/Data/testTildaTrs.xml')
        f.preProc('../Project/tildaDB109.sqlite')
        err = np.sqrt([np.full((4, 5), 250)])
        np.testing.assert_array_equal(f.err, err)
        np.testing.assert_array_equal(f.getSingleSpec(0, -1)[2], err[0][0])
        np.testing.assert_array_equal(f.getArithSpec([0], -1)[2], err[0][0]) # New not tested yet...

    def test_fit_cs(self):
        f = XMLImporter('../Project/Data/testTildaCs.xml')
        f.preProc('../Project/tildaDB109.sqlite')
        spec = Straight()
        fit = SPFitter(spec, f, ([0], 0))
        fit.fit()
        mplplot.plotFit(fit)
        mplplot.show(True)

    def test_fit_trs(self):
        f = XMLImporter('../Project/Data/testTildaTrs.xml')
        f.preProc('../Project/tildaDB109.sqlite')
        spec = Straight()
        fit = SPFitter(spec, f, ([0], 0))
        fit.fit()
        mplplot.plotFit(fit)
        mplplot.show(True)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
