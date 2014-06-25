'''
Created on 31.03.2014

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
# for n in [0,1,2,3]:
#     run = 'Run'+ str(n)
#     print(run + ':')
#     BatchFit.batchFit(Tools.fileList(db,'40_Ca'), db,run)
#     BatchFit.batchFit(Tools.fileList(db,'42_Ca'), db,run)
#     BatchFit.batchFit(Tools.fileList(db,'44_Ca'), db,run)
#     BatchFit.batchFit(Tools.fileList(db,'48_Ca'), db,run)
#      
#     '''Mean of center, sigma and gamma for 40_Ca'''
#     #Analyzer.combineRes('40_Ca', 'gamma',run, db)
#     Analyzer.combineRes('40_Ca', 'sigma',run, db)
#     Analyzer.combineRes('42_Ca', 'sigma',run, db)
#     Analyzer.combineRes('44_Ca', 'sigma',run, db)
#     Analyzer.combineRes('48_Ca', 'sigma',run, db)
#     Analyzer.combineRes('40_Ca', 'center',run, db)
#      
#     '''Calculate the isotope shift to 40_Ca'''
#     Analyzer.combineShift('42_Ca', run, db)
#     Analyzer.combineShift('44_Ca', run, db)
#     Analyzer.combineShift('48_Ca', run, db)
'''what change in the dopplerfrequency results in an angle between the ions and the laser?''' 
con = sqlite3.connect(db)
cur = con.cursor() 
cur.execute('''SELECT frequency FROM Lines WHERE rowid=1''')
(dopplerFreq,) = cur.fetchall()[0]
cur.execute('''SELECT laserFreq FROM Files WHERE type="40_Ca"''')
(laserFreq,)= cur.fetchall()[0]
v = Physics.invRelDoppler(laserFreq, dopplerFreq)
for i in [0,1,3,5,10]:
    x = np.arctan(i/1500)
    print('difference between Ions and Laser of', i, 'mm, results in an angle of:', x*2*np.pi,'deg')
    freqWithoutAngle = Physics.relDoppler(laserFreq, v)
    freqWithAngle = Physics.relDoppler(laserFreq, v*np.cos(x))
    dif = freqWithoutAngle - freqWithAngle
    print('shift in dopplerfrequency:', dif, 'MHz' )