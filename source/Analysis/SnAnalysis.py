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

db = 'V:/Projekte/COLLAPS/Sn/Measurement_and_Analysis_Christian/Sn.sqlite'


'''Plotting spectra'''
# isoL = []
# for i in range(110,137):
#     isoL.append(str(str(i)+'_Sn'))
#     if i == 117 or i == 121 or i == 125 or i == 127 or i == 129 or i == 130 or i == 131:
#         isoL.append(str(str(i)+'_Sn_m'))
# isoL.append('137_Sn_11_2')
# isoL.append('137_Sn_7_2')
# isoL.append('137_Sn_3_2')
# isoL.append('138_Sn')
# Tools.centerPlot(db,isoL)
# print(Physics.wavelenFromFreq(Physics.freqFromWavenumber(22110.525/2)))
# Tools.isoPlot(db, '117_Sn', as_freq=False, laserfreq=Physics.freqFromWavenumber(22110.525), saving=True, show=False, col=True, isom='117_Sn_m')
# Tools.isoPlot(db, '121_Sn', as_freq=False, laserfreq=Physics.freqFromWavenumber(22110.525), saving=True, show=False, col=True, isom='121_Sn_m')
# Tools.isoPlot(db, '125_Sn', as_freq=False, laserfreq=Physics.freqFromWavenumber(22110.525), saving=True, show=False, col=True, isom='125_Sn_m')
# Tools.isoPlot(db, '127_Sn', as_freq=False, laserfreq=Physics.freqFromWavenumber(22110.525), saving=True, show=False, col=True, isom='127_Sn_m')
# Tools.isoPlot(db, '129_Sn', as_freq=False, laserfreq=Physics.freqFromWavenumber(22110.525), saving=True, show=False, col=True, isom='129_Sn_m')
# Tools.isoPlot(db, '130_Sn', as_freq=False, laserfreq=Physics.freqFromWavenumber(22110.525), saving=True, show=False, col=True, isom='130_Sn_m')
# Tools.isoPlot(db, '131_Sn', as_freq=False, laserfreq=Physics.freqFromWavenumber(22110.525), saving=True, show=False, col=True, isom='131_Sn_m')
#
# for i in isoL:
#    Tools.isoPlot(db, i, as_freq=False, laserfreq=Physics.freqFromWavenumber(22110.525), saving=True, show=False, col=True)

'''Crawling'''
# Tools.crawl(db)

'''Fitting the Kepco-Scans!'''
# BatchFit.batchFit(Tools.fileList(db,'Kepco'), db, 'Run2')
# Analyzer.combineRes('Kepco', 'm', 'Run2', db, False)
# Analyzer.combineRes('Kepco', 'b', 'Run2', db, False)

'''Fitting the spectra with Voigt-Fits!'''
#run = 'Run0'
#BatchFit.batchFit(Tools.fileList(db,'120_Sn'), db,run)

'''Mean of center, sigma and gamma for 120_Sn'''
# Analyzer.combineRes('120_Sn', 'gamma',run, db)
# Analyzer.combineRes('120_Sn', 'sigma',run, db)
# Analyzer.combineRes('120_Sn', 'center',run, db, show_plot=True)

'''Calculate the isotope shift to 120_Sn'''
# Analyzer.combineShift('122_Sn', run, db)
