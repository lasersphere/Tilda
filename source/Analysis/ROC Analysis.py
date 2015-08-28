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

db = 'V:/Projekte/COLLAPS/ROC/CaD2.sqlite'

'''Crawling'''
# Tools.crawl(db)
#print(Physics.freqFromWavenumber(12722.2986*2))
spectrum = imp.MCPImporter('V:/Projekte/COLLAPS/ROC/ROCData/Run27_testRun_opticalpumping_Ca48.mcp')
spectrum.preProc(db)
# for i in spectrum.x[0]:
#     print(i)
# print('cts:')
# for i in spectrum.cts[0][0]:
#     print(i)


# spectrum = tldI.TLDImporter('E:/Doktorarbeit/Calcium/2/DataD2/Ca_056.tld')
#plot.plot([spectrum.x[0], spectrum.cts[0][0]])
#plot.show()
'''Fitting the Kepco-Scans!'''
# BatchFit.batchFit(Tools.fileList(db,'Kepco'), db, 'Run0')
# Analyzer.combineRes('Kepco', 'm', 'Run0', db, False)
# Analyzer.combineRes('Kepco', 'b', 'Run0', db, False)

'''Fitting the spectra with Voigt-Fits!'''
run = 'Run0'

interactive = IF.InteractiveFit('Run27_testRun_opticalpumping_Ca48.mcp', db, run)
BatchFit.batchFit(Tools.fileList(db,'40_Ca'), db,run)
BatchFit.batchFit(Tools.fileList(db,'48_Ca'), db,run)
#         BatchFit.batchFit(Tools.fileList(db,'44_Ca'), db,run)
#         BatchFit.batchFit(Tools.fileList(db,'48_Ca'), db,run)
#                      
# #         '''Mean of center, sigma and gamma for 40_Ca'''
# #         Analyzer.combineRes('40_Ca', 'gamma',run, db)
# #         Analyzer.combineRes('40_Ca', 'sigma',run, db)
# #         Analyzer.combineRes('42_Ca', 'sigma',run, db)
# #         Analyzer.combineRes('44_Ca', 'sigma',run, db)
# #         Analyzer.combineRes('48_Ca', 'sigma',run, db)
#         Analyzer.combineRes('40_Ca', 'center',run, db)
#                
'''Calculate the isotope shift to 40_Ca'''
# Analyzer.combineShift('42_Ca', run, db)
# Analyzer.combineShift('44_Ca', run, db)
# Analyzer.combineShift('48_Ca', run, db)

# Tools.isoPlot(db, '43_Ca')