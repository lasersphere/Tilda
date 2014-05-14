'''
Created on 31.03.2014

@author: hammen
'''

from Measurement.SimpleImporter import SimpleImporter
from matplotlib import pyplot as plt

from DBIsotope import DBIsotope
from SPFitter import SPFitter
from Spectra.FullSpec import FullSpec


path = "../test/cd_c_137data.txt"
file = SimpleImporter(path)
iso = DBIsotope('114_Mi-D0', '../test/iso.sqlite')
spec = FullSpec(iso)
#spec = Straight()

fit = SPFitter(spec, file, (0, -1))

fit.fit()

data = file.getSingleSpec(0, -1)

#func = [spec.evaluateE(x, file.laserFreq, file.col, fit.par) for x in data[0]]

#plotdat = spec.toPlotE(file.laserFreq, file.col, fit.par, 100)
plotdat = spec.toPlotE(file.laserFreq, True, fit.par)


#plt.plot(*plotdat)

plt.figure()
plt.plot(plotdat[0], plotdat[1], 'r-')
plt.errorbar(data[0], data[1], yerr = data[2], fmt = 'k^')
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)


#plt.plot(data[0], data[1], 'kp', data[0], func, 'r-')
plt.show()

#Test