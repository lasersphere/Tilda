'''
Created on 31.03.2014

@author: hammen
'''

import os

from Measurement.SimpleImporter import SimpleImporter
from matplotlib import pyplot as plt

from DBIsotope import DBIsotope
from SPFitter import SPFitter
from Spectra.FullSpec import FullSpec
from Spectra.Straight import Straight


path = "../test/cd_c_137data.txt"
file = SimpleImporter(path)
iso = DBIsotope('114_Mi-D0', '../test/iso.sqlite')
spec = FullSpec(iso)
#spec = Straight()

fit = SPFitter(spec, file, (0, -1))

fit.fit()

print(fit.par)

data = file.getSingleSpec(0, -1)
func = [spec.evaluateE(x, file.accVolt, file.col, fit.par) for x in data[0]]


plt.plot(data[0], data[1], 'bp')
#plt.show()
