'''
Created on 05.08.2016

@author: skaufmann
'''
import os, sqlite3, math
from datetime import datetime
import numpy as np

import MPLPlotter as plot
import DBIsotope
from SPFitter import SPFitter
import Spectra.FullSpec as FullSpec
import BatchFit
import Analyzer
import Tools
import Physics
from Measurement.MCPImporter import MCPImporter
from Spectra.FullSpec import FullSpec

import InteractiveFit as IF

db = 'C:\COLLAPS\Online_Analysis_Al\Al\Analysis\Analysis.sqlite'
workdir = os.path.dirname(db)
data_dir = os.path.join(workdir, 'data')

run = 'Run1'

files_26 = Tools.fileList(db, '26_Al')

print(files_26)
meas = [MCPImporter(os.path.join(data_dir, files_26[0]))]

for i, file in enumerate(files_26):
    if i:
        new_meas = MCPImporter(os.path.join(data_dir, files_26[i]))
        if new_meas.getNrSteps(0) == meas[0].getNrSteps(0):  # combine only with same number of steps
            meas.append(new_meas)
            meas[0].cts[0] = np.add(meas[0].cts[0], meas[-1].cts[0])
        else:
            print('could not add file, due to different step number: ', files_26[i])

meas[0].err[0] = np.sqrt(meas[0].cts[0])
# set values that would usually be in the database  or cumment them out and use settings from meas[0]
# meas[0].laserFreq = 757703824
# meas[0].col = True
# meas[0].line = ''
# meas[0].type = '26_Al'
# meas[0].voltDivRatio = "{'accVolt': 10000, 'offset': 1000}"
# meas[0].lineMult = 0.050425
# meas[0].lineOffset = 0.00015

meas[0].preProc(db)
iso = DBIsotope.DBIsotope(db, '26_Al')
isomer = DBIsotope.DBIsotope(db, '26_Al_m')
# isomer = None  # to plot without isomer
spec = FullSpec(iso, isomer)
# print('pars', spec.getPars(), spec.getParNames()[14])

fitter = SPFitter(spec, meas[0], ([4, 5, 6, 7], 0))
# plot.plotFit(fitter)
# plot.show(True)

fitter.fit()
plot.plotFit(fitter)
plot.get_current_axes().legend(['%s ... %s' % (files_26[0], files_26[-1])])

''' plot the isomer with the center from fit result '''
if isomer:
    isomer.center = fitter.par[14]
    spec_isomer = FullSpec(isomer)

    isomer_fit = SPFitter(spec_isomer, meas[0], ([4, 5, 6, 7], 0))
    plot.plotFit(isomer_fit, color='-b')  # just a plot

plot.show(True)
# Create and save graph
# fig = os.path.splitext(path)[0] + run + '.png'
# plot.plotFit(fitter)
# plot.save(fig)
# plot.clear()