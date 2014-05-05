'''
Created on 31.03.2014

@author: hammen
'''

import os

from scipy.optimize import curve_fit

from Measurement.SimpleImporter import SimpleImporter
from matplotlib import pyplot as plt

from DBIsotope import DBIsotope
from SPFitter import SPFitter
from Spectra.Voigt import Voigt
from Spectra.FullSpec import FullSpec


path = "../test/cd_c_137data.txt"
file = SimpleImporter(path)
iso = DBIsotope('114_Mi-D0', '../test/iso.sqlite')
spec = FullSpec(iso, Voigt)

fit = SPFitter(spec, file, (0, -1))

fit.fit()

print(fit.par)


#plt.plot(x, y, 'bp')
#plt.show()

#print(curve_fit(func, x, y, p, err))
