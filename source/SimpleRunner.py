'''
Created on 12.05.2014

@author: hammen, gorges
'''


from Measurement.SimpleImporter import SimpleImporter
from Measurement.KepcoImporterTLD import KepcoImporterTLD
from matplotlib import pyplot as plt

from DBIsotope import DBIsotope
from SPFitter import SPFitter
from Spectra.FullSpec import FullSpec
from Spectra.Straight import Straight

import numpy as np

path = "V:/Projekte/A2-MAINZ-EXP/TRIGA/Measurements and Analysis_Christian/Calcium Isotopieverschiebung/397nm_14_05_13/Daten/KepcoScan_PCI.txt"
file = KepcoImporterTLD(path)
file.type = 'Kepco'
if file.type == 'Kepco':
    spec = Straight()
else:
    iso = DBIsotope(file.type, '../test/iso.sqlite')
    spec = FullSpec(iso)
 
fit = SPFitter(spec, file, (0, -1))
 
fit.fit()
 
data = file.getSingleSpec(0, -1)
plotdat = spec.toPlotE(file.laserFreq, True, fit.par)
 

 
fig = plt.figure(1, (8, 8))
fig.patch.set_facecolor('white')

ax1 = plt.axes([0.1, 0.35, 0.8, 0.6])
plt.errorbar(data[0], data[1], yerr = data[2], fmt = 'k.')
plt.plot(plotdat[0], plotdat[1], 'r-')
ax1.get_xaxis().get_major_formatter().set_useOffset(False)

ax2 = plt.axes([0.1, 0.05, 0.8, 0.25])
plt.errorbar(data[0], fit.calcRes(), yerr = data[2], fmt = 'k.')
ax2.get_xaxis().get_major_formatter().set_useOffset(False)
 
plt.show()


#Test