'''
Created on 31.03.2014

@author: gorges
'''
import os
import sqlite3
import math

import MPLPlotter as plot

from DBIsotope import DBIsotope
from SPFitter import SPFitter
from Spectra.FullSpec import FullSpec
from datetime import datetime
import BatchFit
import Analyzer
import Tools
import Physics

path = 'V:/Projekte/A2-MAINZ-EXP/TRIGA/Measurements and Analysis_Christian/Calcium Isotopieverschiebung/14_06_04_FINAL/'
#db = os.path.join(path, 'CaD1.sqlite')
db = os.path.join(path, 'CaD2.sqlite')


'''Crawling'''
#Tools.crawl(db, 'DataKepco')
#Tools.crawl(db, 'DataD1')
#Tools.crawl(db, 'DataD2')

'''Fitting the Kepco-Scans!'''
#BatchFit.batchFit(Tools.fileList(db,'Kepco'), db, 'Run0')
#Analyzer.combineRes('Kepco', 'm', 'Run0', db, False)
#Analyzer.combineRes('Kepco', 'b', 'Run0', db, False)

'''Fitting the files with Voigt-Fits!'''
run = 'Run0'
BatchFit.batchFit(Tools.fileList(db,'40_Ca'), db,run)
BatchFit.batchFit(Tools.fileList(db,'42_Ca'), db,run)
BatchFit.batchFit(Tools.fileList(db,'44_Ca'), db,run)
BatchFit.batchFit(Tools.fileList(db,'48_Ca'), db,run)

'''Mean of center, sigma and gamma for 40_Ca'''
#Analyzer.combineRes('40_Ca', 'gamma',run, db)
Analyzer.combineRes('40_Ca', 'sigma',run, db)
# Analyzer.combineRes('42_Ca', 'sigma',run, db)
# Analyzer.combineRes('44_Ca', 'sigma',run, db)
# Analyzer.combineRes('48_Ca', 'sigma',run, db)
Analyzer.combineRes('40_Ca', 'center',run, db)

'''Calculate the isotope shift to 40_Ca'''
# Analyzer.combineShift('42_Ca', run, db)
# Analyzer.combineShift('44_Ca', run, db)
# Analyzer.combineShift('48_Ca', run, db)