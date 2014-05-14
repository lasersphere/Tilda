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
plotdat = spec.toPlotE(file.laserFreq, True, fit.par)


fig = plt.figure(1, (8, 8))
fig.patch.set_facecolor('white')

plt.axes([0.1, 0.35, 0.8, 0.6])
plt.plot(plotdat[0], plotdat[1], 'r-')
plt.errorbar(data[0], data[1], yerr = data[2], fmt = 'k.')
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)

plt.axes([0.1, 0.05, 0.8, 0.25])
plt.errorbar(data[0], fit.calcRes(), yerr = data[2], fmt = 'k.')
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)

plt.show()

#Test