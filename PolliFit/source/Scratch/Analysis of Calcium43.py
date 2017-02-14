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
from InteractiveFit import InteractiveFit
import Tools
import Physics

#path = 'V:/Projekte/A2-MAINZ-EXP/TRIGA/Measurements and Analysis_Simon/BuncherMessungen2014/Ca43/'
path = 'C:/Workspace/PolliFit/Data/2016_12_20/'
#db = os.path.join(path, 'CaD1.sqlite')
db = os.path.join(path, 'Ca_Data10kV.sqlite')


'''Crawling'''
#Tools.crawl(db, 'DataKepco')
#Tools.crawl(db, 'Data')
#Tools.crawl(db, 'DataD2')

'''Fitting the Kepco-Scans!'''
# BatchFit.batchFit(Tools.fileList(db,'Kepco'), db, 'Run0')
# Analyzer.combineRes('Kepco', 'm', 'Run0', db, False)
# Analyzer.combineRes('Kepco', 'b', 'Run0', db, False)

'''Fitting the files with Voigt-Fits!'''
for n in [0]:#,1,2,3]:
     run = 'Run'+ str(n)
     print(run + ':')   
# #     BatchFit.batchFit(Tools.fileList(db,'40_Ca'), db,run)
# #     BatchFit.batchFit(Tools.fileList(db,'42_Ca'), db,run)
# #     BatchFit.batchFit(Tools.fileList(db,'44_Ca'), db,run)
# #     BatchFit.batchFit(Tools.fileList(db,'48_Ca'), db,run)
     a = InteractiveFit('2016-04-04_12-29-04_7.bea', db, run)
     BatchFit.batchFit(Tools.fileList(db,'43_Ca'), db,run)
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
# con = sqlite3.connect(db)
# cur = con.cursor() 
# cur.execute('''SELECT frequency FROM Lines''')
# (dopplerFreq,) = cur.fetchall()[0]
# cur.execute('''SELECT laserFreq FROM Files''')
# (laserFreq,)= cur.fetchall()[0]
# v = Physics.invRelDoppler(laserFreq, dopplerFreq)
# for i in [0,1,3,5,10]:
#     x = np.arctan(i/1500)
#     print('change of ionbeam in 1500mm:', i, 'mm. This results in an angle of', x*1000,'mrad between laser and ions')
#     freqWithoutAngle = Physics.dopplerAngle(laserFreq, v, 0)
#     freqWithAngle = Physics.dopplerAngle(laserFreq, v,x)
#     dif = freqWithoutAngle - freqWithAngle
#     print('shift in dopplerfrequency:', dif, 'MHz' )