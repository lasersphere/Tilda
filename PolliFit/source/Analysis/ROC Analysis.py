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

db = 'V:/Projekte/COLLAPS/ROC/ROC_October/CaD2.sqlite'

'''Crawling'''
# Tools.crawl(db)
#print(str(Physics.freqFromWavenumber(12722.2986*2)-Physics.freqFromWavenumber(12722.2984*2)))
#print(Physics.diffDoppler(761906723.3, 30000, 40))
'''Fitting the Kepco-Scans!'''
# BatchFit.batchFit(Tools.fileList(db,'Kepco'), db, 'Run2')
# Analyzer.combineRes('Kepco', 'm', 'Run2', db, False)
# Analyzer.combineRes('Kepco', 'b', 'Run2', db, False)

'''Fitting the spectra with Voigt-Fits!'''
# for i in range(1,3):
    # run = str('Run' + str(i))
    # BatchFit.batchFit(Tools.fileList(db,'40_Ca'), db,run)
    # BatchFit.batchFit(Tools.fileList(db,'42_Ca'), db,run)
    # BatchFit.batchFit(Tools.fileList(db,'44_Ca'), db,run)
    # BatchFit.batchFit(Tools.fileList(db,'48_Ca'), db,run)

#    '''Mean of center, sigma and gamma for 40_Ca'''
# Analyzer.combineRes('40_Ca', 'gamma',run, db)
#     Analyzer.combineRes('40_Ca', 'sigma',run, db)
#     Analyzer.combineRes('42_Ca', 'sigma',run, db)
#     Analyzer.combineRes('44_Ca', 'sigma',run, db)
#     Analyzer.combineRes('48_Ca', 'sigma',run, db)
    # Analyzer.combineRes('40_Ca', 'center',run, db)
    # Analyzer.combineRes('42_Ca', 'center',run, db)
    # Analyzer.combineRes('44_Ca', 'center',run, db)
    # Analyzer.combineRes('48_Ca', 'center',run, db)
    #
    # '''Calculate the isotope shift to 48_Ca'''
    # shift40.append(Analyzer.combineShift('40_Ca', run, db)[2])
    # shift42.append(Analyzer.combineShift('42_Ca', run, db)[2])
    # shift44.append(Analyzer.combineShift('44_Ca', run, db)[2])


# for i in range(11,16):
#     run = str('Run' + str(i))
#     print(run)
#     BatchFit.batchFit(Tools.fileList(db,'52_Ca'), db,run)
#     Analyzer.combineRes('52_Ca','sigma', run, db)
#     Analyzer.combineRes('52_Ca', 'center', run, db)
#     Analyzer.combineShift('52_Ca', run, db)
BatchFit.batchFit(['Run97_ROC_Ca48.mcp'],db,'Run2')