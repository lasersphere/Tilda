'''
Created on 12.05.2014

@author: hammen, gorges
'''
import os

import Measurement.MeasLoad as Meas
from Measurement.SimpleImporter import SimpleImporter
from Measurement.KepcoImporterTLD import KepcoImporterTLD
from Measurement.TLDImporter import TLDImporter
import MPLPlotter as plot

from DBIsotope import DBIsotope
from SPFitter import SPFitter
from Spectra.FullSpec import FullSpec
from Spectra.Straight import Straight

import numpy as np


path = 'V:/Projekte/A2-MAINZ-EXP/TRIGA/Measurements and Analysis_Christian/Calcium Isotopieverschiebung/397nm_14_05_13/Daten/Ca_004.tld'
db = 'V:/Projekte/A2-MAINZ-EXP/TRIGA/Measurements and Analysis_Christian/Calcium Isotopieverschiebung/397nm_14_05_13/AnaDB.sqlite'
file = Meas.load(path, db)

#path = "Z:/Projekte/A2-MAINZ-EXP/TRIGA/Measurements and Analysis_Christian/Calcium Isotopieverschiebung/397nm_14_05_13/Daten/KepcoScan_PCI.txt"
#file = KepcoImporterTLD(path)
#file.type = 'Kepco'
if file.type == 'Kepco':
    spec = Straight()
else:
    #iso = DBIsotope(file.type, file.line,  '../test/iso.sqlite')
    iso = DBIsotope(os.path.join(os.path.dirname(path), '../calciumD1.sqlite'), file.type)
    spec = FullSpec(iso)
 
fit = SPFitter(spec, file, (0, -1))
 
#fit.fit()
 
plot.plotFit(fit)
plot.show()


#Test
