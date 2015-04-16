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

path = 'E:/Doktorarbeit/Calcium/1'
path2 = 'E:/Doktorarbeit/Calcium/2'
db = os.path.join(path, 'CaD2.sqlite')
db2 = os.path.join(path2, 'CaD2.sqlite')

'''Crawling'''
#Tools.crawl(db, 'DataKepco')
#Tools.crawl(db, 'DataD1')
#Tools.crawl(db, 'DataD2')

'''Fitting the Kepco-Scans!'''
# BatchFit.batchFit(Tools.fileList(db,'Kepco'), db, 'Run0')
# Analyzer.combineRes('Kepco', 'm', 'Run0', db, False)
# Analyzer.combineRes('Kepco', 'b', 'Run0', db, False)

'''Fitting the spectra with Voigt-Fits!'''
allshifts42 = []
allshifts44 = []
allshifts48 = []
allshifterrs42 = []
allshifterrs44 = []
allshifterrs48 = []


for x in [path,path2]:
    db = os.path.join(x, 'CaD1.sqlite')
    i = [2,4]
#     i = [1,3]
     
    for n in i:
        run = 'Run'+ str(n)
        if x == path and n == 4:
            break
        print(run + ':')   
        BatchFit.batchFit(Tools.fileList(db,'40_Ca'), db,run)
        BatchFit.batchFit(Tools.fileList(db,'42_Ca'), db,run)
        BatchFit.batchFit(Tools.fileList(db,'44_Ca'), db,run)
        BatchFit.batchFit(Tools.fileList(db,'48_Ca'), db,run)
                     
#         '''Mean of center, sigma and gamma for 40_Ca'''
#         Analyzer.combineRes('40_Ca', 'gamma',run, db)
#         Analyzer.combineRes('40_Ca', 'sigma',run, db)
#         Analyzer.combineRes('42_Ca', 'sigma',run, db)
#         Analyzer.combineRes('44_Ca', 'sigma',run, db)
#         Analyzer.combineRes('48_Ca', 'sigma',run, db)
        Analyzer.combineRes('40_Ca', 'center',run, db)
               
        '''Calculate the isotope shift to 40_Ca'''
        (shifts42, shifterrs42, val42) = Analyzer.combineShift('42_Ca', run, db)
        (shifts44, shifterrs44, val44) = Analyzer.combineShift('44_Ca', run, db)
        (shifts48, shifterrs48, val48) = Analyzer.combineShift('48_Ca', run, db)
           
        allshifts42 += shifts42
        allshifterrs42 += shifterrs42
        allshifts44 += shifts44
        allshifterrs44 += shifterrs44
        allshifts48 += shifts48
        allshifterrs48 += shifterrs48
    
#Tools.isoPlot(db, '43_Ca')

   
print('Done with Combining!')
    
print('Isotopeshifts of 42_Ca:' + str(allshifts42))
print('Isotopeshifterrs of 42_Ca:' + str(allshifterrs42))
val, err, rChi = Analyzer.weightedAverage(allshifts42, allshifterrs42)
err = Analyzer.applyChi(err, rChi)
print('Combined Isotopeshift of 42_Ca:' + str(val) + '(' + str(err) + ')')
    
print('Isotopeshifts of 44_Ca:' + str(allshifts44))
print('Isotopeshifterrs of 44_Ca:' + str(allshifterrs44))
val, err, rChi = Analyzer.weightedAverage(allshifts44, allshifterrs44)
err = Analyzer.applyChi(err, rChi)
print('Combined Isotopeshift of 44_Ca:' + str(val) + '(' + str(err) + ')')
    
print('Isotopeshifts of 48_Ca:' + str(allshifts48))
print('Isotopeshifterrs of 48_Ca:' + str(allshifterrs48))
val, err, rChi = Analyzer.weightedAverage(allshifts48, allshifterrs48)
err = Analyzer.applyChi(err, rChi)
print('Combined Isotopeshift of 48_Ca:' + str(val) + '(' + str(err) + ')')        

# '''what change in the dopplerfrequency results in an angle between the ions and the laser?''' 
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