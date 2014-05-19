'''
Created on 31.03.2014

@author: hammen
'''

from Measurement.SimpleImporter import SimpleImporter
import MPLPlotter as plot

from DBIsotope import DBIsotope
from SPFitter import SPFitter
from Spectra.FullSpec import FullSpec


path = "../test/cd_c_137data.txt"
file = SimpleImporter(path)
iso = DBIsotope('114_Mi', 'Mi-D0',  '../test/iso.sqlite')
spec = FullSpec(iso)

fit = SPFitter(spec, file, (0, -1))

print(fit.spec.parAssign())

fit.fit()


plot.plotFit(fit)
plot.show()

#Test