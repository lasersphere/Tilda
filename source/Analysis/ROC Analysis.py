'''
Created on 31.03.2014

@author: gorges
'''
import os, sqlite3, math
from datetime import datetime
import numpy as np

import MPLPlotter as plot
import DBIsotope
import SPFitter
import Spectra.FullSpec as FullSpec
import BatchFit
import Analyzer
import Tools
import Physics
import Measurement.MCPImporter as imp
import Measurement.TLDImporter as tldI
import InteractiveFit as IF

db = 'V:/Projekte/COLLAPS/ROC/optical/CaD2_optical.sqlite'

'''Crawling'''
#Tools.crawl(db)
#print(Physics.freqFromWavenumber(12722.2986*2))

'''Fitting the Kepco-Scans!'''
BatchFit.batchFit(Tools.fileList(db,'Kepco'), db, 'Run0')
# Analyzer.combineRes('Kepco', 'm', 'Run0', db, False)
# Analyzer.combineRes('Kepco', 'b', 'Run0', db, False)

'''Fitting the spectra with Voigt-Fits!'''
# shift40 = []
# shift42 = []
# shift44 = []
#
# for i in range(0,6):
    # run = str('Run' + str(i))
    # BatchFit.batchFit(Tools.fileList(db,'40_Ca'), db,run)
    # BatchFit.batchFit(Tools.fileList(db,'42_Ca'), db,run)
    # BatchFit.batchFit(Tools.fileList(db,'44_Ca'), db,run)
    # BatchFit.batchFit(Tools.fileList(db,'48_Ca'), db,run)

#    '''Mean of center, sigma and gamma for 40_Ca'''
# Analyzer.combineRes('40_Ca', 'gamma',run, db)
# Analyzer.combineRes('40_Ca', 'sigma',run, db)
# Analyzer.combineRes('42_Ca', 'sigma',run, db)
# Analyzer.combineRes('44_Ca', 'sigma',run, db)
# Analyzer.combineRes('48_Ca', 'sigma',run, db)
#     Analyzer.combineRes('40_Ca', 'center',run, db)
#     Analyzer.combineRes('42_Ca', 'center',run, db)
#     Analyzer.combineRes('44_Ca', 'center',run, db)
#     Analyzer.combineRes('48_Ca', 'center',run, db)
#
#     '''Calculate the isotope shift to 48_Ca'''
#     shift40.append(Analyzer.combineShift('40_Ca', run, db)[2])
#     shift42.append(Analyzer.combineShift('42_Ca', run, db)[2])
#     shift44.append(Analyzer.combineShift('44_Ca', run, db)[2])
#
# print(str(shift40).replace('.',','))
# print(str(shift42).replace('.',','))
# print(str(shift44).replace('.',','))