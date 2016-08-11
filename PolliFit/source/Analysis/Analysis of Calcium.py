'''
Created on 04.08.2015

@author: gorges
'''
import os
import sqlite3
import math

import MPLPlotter as plot
import numpy as np

from DBIsotope import DBIsotope
from SPFitter import SPFitter
from Spectra.FullSpec import FullSpec
from datetime import datetime
import BatchFit
import Analyzer
import Tools
import Physics

db = 'V:/Projekte/COLLAPS/ROC/CaD2.sqlite'

'''Plotting the Isotopes'''
isoL = ['40_Ca', '43_Ca', '44_Ca']

Tools.centerPlot(db, isoL)

# for i in range(40,48,2):
#     iso = str(str(i) + '_Ca')
#     isoL.append(iso)
# Tools.isoPlot(db, '51_Ca')
#Tools.centerPlot(db, ['48_Ca'])
#print(Physics.wavenumber(762809833.7407918))
#print(Physics.freqFromWavenumber(30))
'''Crawling'''
# #Tools.crawl(db, 'DataKepco')
# Tools.crawl(db, 'DataD1')
# Tools.crawl(db, 'DataD2')

'''Fitting the Kepco-Scans!'''
# BatchFit.batchFit(Tools.fileList(db,'Kepco'), db, 'Run0')
# Analyzer.combineRes('Kepco', 'm', 'Run0', db, False)
# Analyzer.combineRes('Kepco', 'b', 'Run0', db, False)

'''Fitting the spectra with Voigt-Fits!'''
#         BatchFit.batchFit(Tools.fileList(db,'40_Ca'), db,run)
#         BatchFit.batchFit(Tools.fileList(db,'42_Ca'), db,run)
#         BatchFit.batchFit(Tools.fileList(db,'44_Ca'), db,run)
#         BatchFit.batchFit(Tools.fileList(db,'48_Ca'), db,run)
#                      
#         '''Mean of center, sigma and gamma for 40_Ca'''
#         Analyzer.combineRes('40_Ca', 'gamma',run, db)
#         Analyzer.combineRes('40_Ca', 'sigma',run, db)
#         Analyzer.combineRes('42_Ca', 'sigma',run, db)
#         Analyzer.combineRes('44_Ca', 'sigma',run, db)
#         Analyzer.combineRes('48_Ca', 'sigma',run, db)
#        Analyzer.combineRes('40_Ca', 'center',run, db)