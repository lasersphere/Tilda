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

import InteractiveFit as IF

db = 'C:/Users/Christian/Dropbox/Measurement_and_Analysis_Christian/Sn.sqlite'


'''Plotting spectra'''
isoL = []
for i in range(114,133):
    isoL.append(str(str(i)+'_Sn'))
    # if i == 117 or i == 121 or i == 125 or i == 127 or i == 129 or i == 130 or i == 131:
    #    isoL.append(str(str(i)+'_Sn_m'))

isoL.append('112_Sn')

# Tools.centerPlot(db,isoL)
wavenumber = 22110.525
print(Physics.freqFromWavenumber(wavenumber))
print(Physics.wavelenFromFreq(Physics.freqFromWavenumber(wavenumber/2)))
# for i in isoL:
#    Tools.isoPlot(db, i, as_freq=False, laserfreq=Physics.freqFromWavenumber(wavenumber),
#                  saving=True, show=False, col=True)
# Tools.isoPlot(db, '117_Sn', as_freq=False, laserfreq=Physics.freqFromWavenumber(wavenumber),
#               saving=True, show=True, col=True, isom_name='117_Sn_m')
# Tools.isoPlot(db, '121_Sn', as_freq=False, laserfreq=Physics.freqFromWavenumber(wavenumber),
#               saving=True, show=False, col=True, isom_name='121_Sn_m')
# Tools.isoPlot(db, '125_Sn', as_freq=False, laserfreq=Physics.freqFromWavenumber(wavenumber),
#               saving=True, show=False, col=True, isom_name='125_Sn_m')
# Tools.isoPlot(db, '127_Sn', as_freq=False, laserfreq=Physics.freqFromWavenumber(wavenumber),
#               saving=True, show=False, col=True, isom_name='127_Sn_m')
# Tools.isoPlot(db, '129_Sn', as_freq=False, laserfreq=Physics.freqFromWavenumber(wavenumber),
#               saving=True, show=False, col=True, isom_name='129_Sn_m')
# Tools.isoPlot(db, '130_Sn', as_freq=False, laserfreq=Physics.freqFromWavenumber(wavenumber),
#               saving=True, show=False, col=True, isom_name='130_Sn_m')
# Tools.isoPlot(db, '131_Sn', as_freq=False, laserfreq=Physics.freqFromWavenumber(wavenumber),
#               saving=True, show=False, col=True, isom_name='131_Sn_m')

'''Crawling'''
Tools.crawl(db)
con = sqlite3.connect(db)
cur = con.cursor()
cur.execute('''UPDATE Files SET laserFreq=662856863.742045, voltDivRatio="{'accVolt':10000, 'offset':1000}", lineMult=0.050414, lineOffset=0''')
con.commit()
con.close()

'''Fitting the Kepco-Scans!'''
# for i in range(0,2):
#     run = 'ZKepRun' + str(i)
#     BatchFit.batchFit(Tools.fileList(db,'Kepco'), db, run)
#     Analyzer.combineRes('Kepco', 'm', run, db, show_plot=False)
#     Analyzer.combineRes('Kepco', 'b', run, db, show_plot=False)

'''Fitting the spectra with Voigt-Fits!'''
for j in isoL:
    for i in range(0,1):
        run = 'Run' + str(i)
        BatchFit.batchFit(Tools.fileList(db,j), db,run)

        '''Mean of center, sigma and gamma for the isotopes'''
        Analyzer.combineRes(j, 'center',run, db)
        # Analyzer.combineRes(j, 'sigma',run, db)
        #Analyzer.combineRes(j, 'gamma',run, db)
#
#     '''Calculate the isotope shift to 124_Sn'''
#     # Analyzer.combineShift('122_Sn', run, db)
