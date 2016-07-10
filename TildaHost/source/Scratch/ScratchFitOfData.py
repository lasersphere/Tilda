"""
author: Simon Kaufmann

created on 02_07_16
simple fit script to fit a dataset of 3 points with a straight
"""

import MPLPlotter
from Measurement.SpecData import SpecData
from SPFitter import SPFitter
from Spectra import Straight

data = [[69.22, 92.7, 116.12], [8, 32, 56]]
spec_data = SpecData()
spec_data.x = [data[1]]
spec_data.cts = [[data[0]]]
spec_data.err = [[[i * 0.001 for i in data[0]]]]
spec_data.laserFreq = 1
# spec_data.offset = 1
spec_data.col = 1
spec_data.nrScalers = [1]
fit = SPFitter(Straight.Straight(), spec_data, ([0], -1))
fit.fit()
print(fit.result())
# fit.spec.evaluate(9, [60, 3])
fit.spec.x_min = 7
fit.spec.x_max = 57
MPLPlotter.plotFit(fit)
MPLPlotter.show()
print(spec_data.err)