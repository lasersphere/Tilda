"""
author: Simon Kaufmann

created on 02_07_16
simple fit script to fit a dataset of 3 points with a straight
"""

import os

import numpy as np

from Tilda.PolliFit import MPLPlotter
from Tilda.PolliFit.Measurement.SpecData import SpecData
from Tilda.PolliFit.SPFitter import SPFitter
from Tilda.PolliFit.Spectra import Gauss
from Tilda.PolliFit.Spectra import Straight

nau_data = [[0, 32, 63.5], [26.17, 52.91, 79.16]]

workdir = 'C:\\Users\\Simon-K\\Documents\\FPraktikum\\Auswertungen\\NaumannBraun'
files = os.listdir(workdir)
fit_results = {}
for file in files:
    try:
        if file.split('.')[1] == 'Spe':
            print('working on : ', file)
            with open(os.path.join(workdir, file)) as f:
                file_as_str = str(f.read().replace('\n', '').replace('\"', ''))
                file_as_str = file_as_str[file_as_str.find('DATA:') + 5:]
                file_as_str = file_as_str[:file_as_str.find('$ROI:')]
                data = np.fromstring(file_as_str, sep='\t')[2:]  # drop length info

                spec_data = SpecData()
                spec_data.x = [np.arange(0, len(data), 1)]
                spec_data.cts = [[data]]
                err = np.sqrt(data)
                err[err == 0] = 1
                spec_data.err = [[err]]
                spec_data.laserFreq = 1
                # spec_data.offset = 1
                spec_data.col = 1
                spec_data.nrScalers = [1]
                fit = SPFitter(Gauss.Gauss(), spec_data, ([0], -1))

                fit.fit()
                fit_results[file] = fit.result()
                # fit.spec.evaluate(9, [60, 3])
                # fit.spec.x_min = 7
                # fit.spec.x_max = 57
                # MPLPlotter.plotFit(fit)
                # MPLPlotter.show()
                print(spec_data.err)

    except Exception as e:
        print(e)


delays = [16, 32, 4, 8, 63.5, 0]
fit_res = [44.6, 60.57, 32.60, 36.82, 91.38, 28.55]
data = [delays, fit_res]
# data = nau_data
spec_data = SpecData()
spec_data.x = [data[1]]
spec_data.cts = [[data[0]]]
spec_data.err = [[[1 for i in data[0]]]]
spec_data.laserFreq = 1
# spec_data.offset = 1
spec_data.col = 1
spec_data.nrScalers = [1]
fit = SPFitter(Straight.Straight(), spec_data, ([0], -1))
fit.fit()
print(fit.result())
# fit.spec.evaluate(9, [60, 3])
fit.spec.x_min = 0
fit.spec.x_max = 100
MPLPlotter.plotFit(fit)
MPLPlotter.get_current_axes().set_xlabel('channels')
MPLPlotter.get_current_axes().set_ylabel('delay_set [ns]')
slope = fit.result()[0][1]['m'][0]
offset = fit.result()[0][1]['b'][0]
# MPLPlotter.show()

for key, val in fit_results.items():
    print('---------------------')
    print('fit resulst of ', key)
    print('sigma', val[0][1]['sigma'], 'sigma [ns]:', val[0][1]['sigma'][0] * slope)
    print('mu', val[0][1]['mu'], 'mu [ns]:', val[0][1]['mu'][0] * slope + offset)

# print(spec_data.err)

