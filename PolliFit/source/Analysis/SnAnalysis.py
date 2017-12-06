'''
Created on 11.07.2016

@author: gorges
'''

import os, sqlite3, math
from datetime import datetime
import numpy as np

import time
import MPLPlotter as plot
import DBIsotope
import SPFitter
import Spectra.FullSpec as FullSpec
import BatchFit
import Analyzer
import Tools
import Physics
from KingFitter import KingFitter

import InteractiveFit as IF

db = 'C:/Users/Christian/Downloads/Sn.sqlite'

'''preparing a list of isotopes'''
# isoL = ['109_Sn','112_Sn']
# for i in range(114,121, 1):
#     isoL.append(str(str(i)+'_Sn'))
# isoL.append('122_Sn')
isoL = ['109_Sn']
for i in range(112,135, 1):
    isoL.append(str(str(i)+'_Sn'))
    if i == 113 or i == 117 or i == 119 or i == 121 or i == 123 or i == 125\
            or i == 127 or i == 128 or i == 129 or i == 130 or i == 131:
        isoL.append(str(str(i)+'_Sn_m'))
# freq = 662305065
# isoL = ['112_Sn','114_Sn','115_Sn','116_Sn','118_Sn','119_Sn','120_Sn','122_Sn',
# '125_Sn','126_Sn','128_Sn','131_Sn','132_Sn','133_Sn','134_Sn']

'''Plotting spectra'''
# Tools.centerPlot(db,isoL)
# Tools.isoPlot(db, '131_Sn', isovar='_m')
# Tools.isoPlot(db, '131_Sn')
# Tools.isoPlot(db, '132_Sn')
# Tools.isoPlot(db, '109_Sn')
# wavenumber = 22112.96
# print(Physics.freqFromWavenumber(wavenumber))
# print(Physics.wavelenFromFreq(Physics.freqFromWavenumber(wavenumber)))

# for i in isoL:
#    Tools.isoPlot(db, i, as_freq=False, laserfreq=Physics.freqFromWavenumber(wavenumber),
#                  saving=True, show=False, col=True)
#Tools.isoPlot(db, '117_Sn', as_freq=False, laserfreq=Physics.freqFromWavenumber(wavenumber),
#              saving=True, show=True, col=True, isom_name='117_Sn_m')
#Tools.isoPlot(db, '119_Sn', as_freq=False, laserfreq=Physics.freqFromWavenumber(wavenumber),
#              saving=True, show=True, col=True, isom_name='119_Sn_m')
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

# '''Crawling'''
# # Tools.crawl(db)
# chiSqs = [[]]
# chiSqs2 = [[]]
# divratios = []
# accratios = []
# l = 0
# ind = 0
# # print(Physics.diffDoppler(662378005, 40000, 124))
#
# for k in range(99891, 99892, 1): #
#     k=k/10
#     acc = k
#
#     for i in range(100003, 100004, 1): #
#         freq = 662378005 +(10000-acc)*27.5
#         #i = i+ind
#         i = i/100
#         con = sqlite3.connect(db)
#         cur = con.cursor()
#         cur.execute('''UPDATE Files SET laserFreq=662929863.205568,
#           voltDivRatio="{'accVolt':9997.1, 'offset':1000.85}",
#           lineMult=0.050425, lineOffset=0.00015''')
#         divratio = str(''' voltDivRatio="{'accVolt':''' + str(k) + ''', 'offset':''' + str(i) + '''}" ''')
#         cur.execute('''UPDATE Files SET ''' + divratio)
#
#         cur.execute('''UPDATE Lines SET frequency='''+str(freq))
#
#         con.commit()
#         con.close()
#         divratios.append(i)
#         accratios.append(k)
# #
'''Fitting the Kepco-Scans!'''
# for i in range(0,1):
#     run = 'ZKepRun' + str(i)
#     BatchFit.batchFit(Tools.fileList(db,'Kepco'), db, run)
#     # Analyzer.combineRes('Kepco', 'm', run, db, show_plot=False)
#     # Analyzer.combineRes('Kepco', 'b', run, db, show_plot=False)
# BatchFit.batchFit(Tools.fileList(db, '124_Sn'), db, 'Run10')
#         '''Fitting the spectra with Voigt-Fits!'''
#         BatchFit.batchFit(Tools.fileList(db,'124_Sn'), db,'Run0')
#         # BatchFit.batchFit(['124Sn_no_protonTrigger_Run018.mcp','124Sn_no_protonTrigger_Run021.mcp','124Sn_no_protonTrigger_Run024.mcp',
#         #                    '124Sn_no_protonTrigger_Run025.mcp','124Sn_no_protonTrigger_Run031.mcp','124Sn_no_protonTrigger_Run032.mcp',
#         #                    '124Sn_no_protonTrigger_Run037.mcp','124Sn_no_protonTrigger_Run038.mcp','124Sn_no_protonTrigger_Run043.mcp',
#         #                    '124Sn_no_protonTrigger_Run054.mcp','124Sn_no_protonTrigger_Run056.mcp','124Sn_no_protonTrigger_Run057.mcp',
#         #                    '124Sn_no_protonTrigger_Run061.mcp','124Sn_no_protonTrigger_Run138.mcp','124Sn_no_protonTrigger_Run139.mcp',
#         #                    '124Sn_no_protonTrigger_Run143.mcp','124Sn_no_protonTrigger_Run152.mcp','124Sn_no_protonTrigger_Run154.mcp',
#         #                    '124Sn_no_protonTrigger_Run175.mcp','124Sn_no_protonTrigger_Run176.mcp','124Sn_no_protonTrigger_Run182.mcp',
#         #                    '124Sn_no_protonTrigger_Run195.mcp','124Sn_no_protonTrigger_Run204.mcp'], db, 'Run0')
#         # Analyzer.combineRes('124_Sn', 'sigma', 'Run0', db)
#         # Analyzer.combineRes('124_Sn', 'center', 'Run0', db)
#         isotopeShifts = []
#         isotopeShiftErrs = []
#         for j in isoL:
#             '''if there is an isomer at the isotope, we need Run1:'''
#             if j == '113_Sn' or j == '121_Sn' or  j == '123_Sn' or j == '125_Sn'\
#                     or j == '129_Sn' or j == '130_Sn' or j == '131_Sn':
#                 run = 'Run1'
#             elif j == '127_Sn':
#                 run = 'Run2'
#             else:
#                 run = 'Run0'
#
#             if j == '117_Sn':
#                 con = sqlite3.connect(db)
#                 cur = con.cursor()
#                 cur.execute('''DELETE FROM FitRes WHERE iso="117_Sn"''')
#                 con.commit()
#                 con.close()
#                 BatchFit.batchFit(['117Sn_no_protonTrigger_Run141.mcp','117Sn_no_protonTrigger_Run142.mcp',
#                                    '117Sn_no_protonTrigger_Run179.mcp'], db, 'Run1')
#                 BatchFit.batchFit(['117Sn_no_protonTrigger_Run033.mcp','117Sn_no_protonTrigger_Run034.mcp'], db, run)
#                 con = sqlite3.connect(db)
#                 cur = con.cursor()
#                 cur.execute('''UPDATE FitRes SET run='Run10' WHERE iso="117_Sn"''')
#                 con.commit()
#                 con.close()
#
#                 shift = Analyzer.combineShift(j, 'Run10', db)
#                 isotopeShifts.append(shift[2])
#                 isotopeShiftErrs.append(np.sqrt(np.square(shift[3])))
#                 Analyzer.combineShift(str(j)+'_m', 'Run1', db)
#                 Analyzer.combineRes(j, 'Au', 'Run10', db)
#                 Analyzer.combineRes(str(j)+'_m', 'Au', 'Run1', db)
#                 Analyzer.combineRes(str(j)+'_m', 'Bu', 'Run1', db)
#
#             elif j == '119_Sn':
#                 con = sqlite3.connect(db)
#                 cur = con.cursor()
#                 cur.execute('''DELETE FROM FitRes WHERE iso="119_Sn"''')
#                 con.commit()
#                 con.close()
#                 BatchFit.batchFit(['119Sn_no_protonTrigger_Run137.mcp','119Sn_no_protonTrigger_Run174.mcp'], db, 'Run1')
#                 BatchFit.batchFit(['119Sn_no_protonTrigger_Run029.mcp','119Sn_no_protonTrigger_Run030.mcp',
#                                    '119Sn_no_protonTrigger_Run055.mcp', '119Sn_no_protonTrigger_Run153.mcp'], db, run)
#                 con = sqlite3.connect(db)
#                 cur = con.cursor()
#                 cur.execute('''UPDATE FitRes SET run='Run10' WHERE iso="119_Sn"''')
#                 con.commit()
#                 con.close()
#                 shift = Analyzer.combineShift(j, 'Run10', db)
#                 isotopeShifts.append(shift[2])
#                 isotopeShiftErrs.append(np.sqrt(np.square(shift[3])))
#                 Analyzer.combineShift(str(j)+'_m', 'Run1', db)
#                 Analyzer.combineRes(j, 'Au', 'Run10', db)
#                 Analyzer.combineRes(str(j)+'_m', 'Au', 'Run1', db)
#                 Analyzer.combineRes(str(j)+'_m', 'Bu', 'Run1', db)
#             elif j != '110_Sn' or j != '111_Sn':
#                 BatchFit.batchFit(Tools.fileList(db,j), db,run)
#                 '''Calculate the isotope shift to 124_Sn'''
#                 shift = Analyzer.combineShift(j, run, db)
#                 isotopeShifts.append(shift[2])
#                 isotopeShiftErrs.append(np.sqrt(np.square(shift[3])))
#                 if run == 'Run1' or run == 'Run2':
#                     shift_m = Analyzer.combineShift(str(j)+'_m', run, db)
#                     if j != '130_Sn' and j != '129_Sn':
#                         Analyzer.combineRes(j, 'Au', run, db)
#                     if j == '121_Sn' or j == '123_Sn' or j == '125_Sn' or j == '131_Sn' or j == '127_Sn':
#                         Analyzer.combineRes(j, 'Bu', run, db)
#                     if j != '127_Sn':
#                         Analyzer.combineRes(str(j)+'_m', 'Au', run, db)
#                         Analyzer.combineRes(str(j)+'_m', 'Bu', run, db)
#                 elif j == '109_Sn' or j == '133_Sn'  or j == '115_Sn':
#                     Analyzer.combineRes(j, 'Au', run, db)
#                     if j!= '115_Sn':
#                         Analyzer.combineRes(j, 'Bu', run, db)
#                 '''Mean of center, sigma and gamma for the isotopes'''
#                 # Analyzer.combineRes(j, 'center',run, db, show_plot=False)
#                 # Analyzer.combineRes(j, 'sigma',run, db, show_plot=False)
#                 '''comparison if intensities'''
#                 # Ints0 = Analyzer.combineRes(j, 'Int0',run, db)
#                 # Ints1 = Analyzer.combineRes(j, 'Int1',run, db)
#                 # ints = []
#                 # for i,j in enumerate(Ints0[3][1]):
#                 #     ints.append(float(Ints1[3][1][i]/j))
#                 # print(ints)
#         '''calculating red. Chi^2'''
#         litvals = [-1898.6, -1380.3, -1115.7, -1044.3, -842.6, -759.1, -588.4, -521.4, -360.3, -162.8]
#         literrs = [18.7, 8.9, 8.3, 8.3, 8, 8.1, 8.3, 8.3, 8.9, 10]
#         chisq = 0
#         chisq2 = 0
#         for index in range(0, len(isoL)):
#             chisq += np.square((isotopeShifts[index]-litvals[index])/(np.sqrt(np.square(isotopeShiftErrs[index])+np.square(literrs[index]))))
#             chisq2 += np.square((isotopeShifts[index]-litvals[index])/(np.sqrt(np.square(isotopeShiftErrs[index]))))
#         chiSqs[l].append(float(chisq))
#         chiSqs2[l].append(float(chisq2))
#         print('red Chi^2:', chiSqs[l])
#         print(k)
#     print(k)
#     l+=1
#     ind+=1
#     chiSqs.append([])
#     chiSqs2.append([])
# strA = str(accratios).replace(',','\t')
# strA = strA.replace('.', ',')
# print(strA[1:-1])
# strA = str(divratios).replace(',','\t')
# strA = strA.replace('.', ',')
# print(strA[1:-1])
# for i in chiSqs:
#     strI = str(i).replace(',','\t')
#     strI = strI.replace('.', ',')
#     print(strI[1:-1])
# for i in chiSqs2:
#     strI = str(i).replace(',','\t')
#     strI = strI.replace('.', ',')
#     print(strI[1:-1])

# j = '117_Sn'
# con = sqlite3.connect(db)
# cur = con.cursor()
# cur.execute('''DELETE FROM FitRes WHERE iso="117_Sn"''')
# con.commit()
# con.close()
# BatchFit.batchFit(['117Sn_no_protonTrigger_Run141.mcp','117Sn_no_protonTrigger_Run142.mcp',
#                    '117Sn_no_protonTrigger_Run179.mcp'], db, 'Run1')
# BatchFit.batchFit(['117Sn_no_protonTrigger_Run033.mcp','117Sn_no_protonTrigger_Run034.mcp'], db, 'Run0')
# con = sqlite3.connect(db)
# cur = con.cursor()
# cur.execute('''UPDATE FitRes SET run='Run10' WHERE iso="117_Sn"''')
# con.commit()
# con.close()
#
# shift = Analyzer.combineShift(j, 'Run10', db)
# Analyzer.combineShift(str(j)+'_m', 'Run1', db)
# Analyzer.combineRes(j, 'Au', 'Run10', db)
# # Analyzer.combineRes(str(j)+'_m', 'Au', 'Run1', db)
# Analyzer.combineRes(str(j)+'_m', 'Bu', 'Run1', db)

'''performing a King fit analysis'''
litvals = {'112_Sn':[-0.748025649,.0077],
            '114_Sn':[-0.601624554,.0077],
           '116_Sn':[ -0.464108311,.0077],
            '117_Sn':[-0.422258642,.0075],
           '118_Sn':[-0.327818629,.0077],
           '119_Sn':[-0.303343067,.0075],
            '120_Sn':[-0.202198458,.0080],
            '122_Sn':[-0.093007073,.0077]}#Fricke charge radii

# litvals = {'112_Sn':[-1659.44,0.21],'113_Sn':[-1520.45,2.30],'114_Sn':[-1341.83, 0.21],'115_Sn':[-1246.07,0.19],
#             '116_Sn':[-1017.19,0.21],'117_Sn':[-912.58,0.19],'118_Sn':[-711.39,0.21],'119_Sn':[-620.74,0.19],
#            '120_Sn':[-441.15,0.15],'121_Sn':[-350.35,2.70],'122_Sn':[-205.8,0.21],'123_Sn':[-143.95,2.01],
#            '125_Sn':[63.15,7.5],'117_Sn_m':[-924.45,1.81],'121_Sn_m':[-369.15,1.71]}#Anselment isotope shift
#
# litvals = {'109_Sn' :[ -1.016 ,0.016],'112_Sn' :[ -0.753 ,0.007],'113_Sn' :[ -0.714 ,0.01],'113_Sn_m' :[ -0.709 ,0.009],
#            '114_Sn' :[ -0.609 ,0.005],'115_Sn' :[ -0.564 ,0.007],'116_Sn' :[ -0.464 ,0.004],'117_Sn' :[ -0.414 ,0.004],
#            '117_Sn_m' :[ -0.42 ,0.004],'118_Sn' :[ -0.326 ,0.005],'119_Sn' :[ -0.289 ,0.004],'119_Sn_m' :[ -0.304 ,0.005],
#            '120_Sn' :[ -0.203 ,0.005],'121_Sn' :[ -0.145 ,0.005],'121_Sn_m' :[ -0.176 ,0.002],'122_Sn' :[ -0.096 ,0.003],
#            '123_Sn' :[ -0.061 ,0.001],'123_Sn_m' :[ -0.074 ,0.004],'125_Sn' :[ 0.032 ,0.004],'125_Sn_m' :[ 0.018 ,0.007],
#            '126_Sn' :[ 0.086 ,0.004],'127_Sn' :[ 0.116 ,0.009],'127_Sn_m' :[ 0.104 ,0.013],'128_Sn' :[ 0.163 ,0.01],
#            '129_Sn' :[ 0.172 ,0.018],'129_Sn_m' :[ 0.196 ,0.013],'130_Sn' :[ 0.234 ,0.015],'130_Sn_m' :[ 0.198 ,0.022],
#             '131_Sn' :[ 0.227 ,0.026],'131_Sn_m' :[ 0.27 ,0.018],'132_Sn' :[ 0.3 ,0.022],'133_Sn' :[ 0.375 ,0.017],
#            '134_Sn' :[ 0.532 ,0.006]}

# BatchFit.batchFit(['121Sn_no_protonTrigger_Run123.mcp'],db, 'Run1')
# Analyzer.combineShift('128_Sn', 'Run1', db)
# Analyzer.combineShift('119_Sn', 'Run1', db)
# Analyzer.combineRes('128_Sn_m', 'Bu', 'Run1', db)
# Analyzer.combineShift('109_Sn', 'Run0', db)
# Analyzer.combineShift('125_Sn_m', 'Run1', db)
# Analyzer.combineRes('125_Sn_m', 'Bu', 'Run1', db)
# Analyzer.combineRes('125_Sn', 'Au', 'Run1', db)
# Analyzer.combineRes('125_Sn', 'Bu', 'Run1', db)
# Analyzer.combineRes('121_Sn', 'Au', 'Run1', db)

# king = KingFitter(db, showing=False, litvals=litvals)
# run = -1
# # king.kingFit(alpha=849, findBestAlpha=True, run=run)
# king.calcChargeRadii(isotopes=isoL, run=run)

'''producing a LaTeX-table'''
# for i in isoL:
#     a = str(i)
#     for char in "_Sn":
#         a = a.replace(char, "")
#     con = sqlite3.connect(db)
#     cur = con.cursor()
#     cur.execute('''SELECT I FROM Isotopes WHERE iso=?''', (i,))
#     nuclearSpin = int(cur.fetchall()[0][0]*2)
#     if nuclearSpin == 0:
#         nuclearSpin = str(nuclearSpin)
#     else:
#         nuclearSpin = str(nuclearSpin) + '/2'
#     cur.execute('''SELECT val, statErr, systErr FROM Combined WHERE parname=? AND iso = ?''', ('shift',i))
#     (shiftval, shiftstatErr, shiftsystErr) = cur.fetchall()[0]
#     shiftval = '%.1f' % (shiftval)
#     shiftstatErr = int(np.round(shiftstatErr*10,0))
#     shiftsystErr = int(np.round(shiftsystErr*10,0))
#     if a != '124':
#         cur.execute('''SELECT val, statErr, systErr FROM Combined WHERE parname=? AND iso = ?''', ('delta_r_square',i))
#         (crval, crstatErr, crsystErr) = cur.fetchall()[0]
#         crval = '%.4f' % (crval)
#         crstatErr = int(np.round(crstatErr*10000,0))
#         crsystErr = int(np.round(crsystErr*10000,0))
#     else:
#         crval = 0
#         crstatErr = 0
#         crsystErr = 0
#     output = str('$' + a + '$' + ' & $'+ str(nuclearSpin) + '$ & $' + str(shiftval) + '(' + str(shiftstatErr) + ') (' + str(shiftsystErr) + ')$ & $' +
#           str(crval) + '(' + str(crstatErr) + ') (' + str(crsystErr) + ')$' + str('\\')+str('\\'))
#     # output = str('$' + a + '$' + ' & $'+ str(nuclearSpin) + '$ & $' + str(shiftval) + '(' + str(shiftstatErr) + ') $' + str('\\') + str('\\'))
#     output = output.replace('.', ',')
#     print(output)
# isoL = ['117_Sn_m']
#
# '''for A and B, mu and Q'''
# muRef = -1.00104
# dMuRef = 0.00007
# aRef = -247.6
# daRef = 2
# eV = 486
# deV = 20
# for i in isoL:
#     a = str(i)
#     for char in "_Sn":
#         a = a.replace(char, "")
#     con = sqlite3.connect(db)
#     cur = con.cursor()
#     cur.execute('''SELECT I FROM Isotopes WHERE iso=?''', (i,))
#     nuclearSpin = int(cur.fetchall()[0][0]*2)
#     nuclSpin = float(nuclearSpin)
#     if nuclearSpin == 0:
#         nuclearSpin = str(nuclearSpin)
#     else:
#         nuclearSpin = str(nuclearSpin) + '/2'
#     avalStr = ''
#     muvalStr = ''
#     bvalStr = ''
#     qValStr = ''
#     printing = False
#     cur.execute('''SELECT val, statErr, systErr FROM Combined WHERE parname=? AND iso = ?''', ('Au',i))
#     try:
#         (aval, astatErr, asystErr) = cur.fetchall()[0]
#         mu = float(aval) * nuclSpin * muRef/aRef
#         dstatmu = float(astatErr) * nuclSpin * muRef/aRef
#         deltaMu = np.sqrt((float(asystErr) * nuclSpin * muRef/aRef)**2 + (float(aval) * nuclSpin * dMuRef/aRef)**2 +
#                           (float(aval) * nuclSpin * muRef * daRef/(aRef**2))**2 )
#
#         aval = '%.2f' % aval
#         astatErr = int(np.round(astatErr*100,0))
#         asystErr = int(np.round(asystErr*100,0))
#         avalStr = '$' + str(aval) + '(' + str(astatErr) + ') (' + str(asystErr) + ') $'
#         mu = '%.3f' % mu
#         muerr = np.sqrt(dstatmu**2+deltaMu**2)
#         dstatmu = int(np.round(dstatmu*1000,0))
#         deltaMu = int(np.round(deltaMu*1000,0))
#         muvalStr = '$' + str(mu) + '(' + str(dstatmu) + ') (' + str(deltaMu) + ') $'
#         # muvalStr = str(mu) + '\t' + '%.5f' % (muerr)
#         cur.execute('''SELECT val, statErr, systErr FROM Combined WHERE parname=? AND iso = ?''', ('Bu', i))
#         printing = True
#         try:
#             (bval, bstatErr, bsystErr) = cur.fetchall()[0]
#             q = bval/eV
#             dstatq = bstatErr/eV
#             deltaQ = np.sqrt((bsystErr/eV)**2 + (deV * bval/(eV**2))**2)
#             bval = '%.2f' % (bval)
#             bstatErr = int(np.round(bstatErr*100,0))
#             bsystErr = int(np.round(bsystErr*100,0))
#             bvalStr = str('$' + str(bval) + '(' + str(bstatErr) + ') (' + str(bsystErr)  + ')$')
#             q = '%.4f' % (q)
#             qerr = np.sqrt(dstatq**2+deltaQ**2)
#             dstatq = int(np.round(dstatq*10000,0))
#             deltaQ = int(np.round(deltaQ*10000,0))
#             qValStr = str('$' + str(q) + '(' + str(dstatq) + ') (' + str(deltaQ)  + ')$')
#             # qValStr = str(q) + '\t' + '%.5f' % (qerr)
#         except:
#             bvalStr = ''
#             pass
#     except:
#         pass
#
#     finally:
#         if printing:
#             output = str('$' + a + '$' + ' & $'+ str(nuclearSpin) + '$ &' +avalStr + '& ' + muvalStr +'& ' + bvalStr
#                          +'& ' + qValStr + str('\\')+str('\\'))
#             # output = str('$' + a + '$' + ' & $'+ str(nuclearSpin) + '$ &' +avalStr + '& ' + bvalStr + str('\\')+ str('\\') )
#             # output = str(a + '\t' + str(nuclearSpin) + '\t' +avalStr + '\t' + muvalStr +'\t' + bvalStr
#             #              +'\t' + qValStr)
#             output = output.replace('.', ',')
#             print(output)


'''selecting Au, Bu, delta r^2, ...'''
# listAuBu = ['Au', 'Bu', 'delta_r_square', 'shift']
# for j in listAuBu:
#     print(j)
#     con = sqlite3.connect(db)
#     cur = con.cursor()
#     cur.execute('''SELECT iso, val, statErr, systErr FROM Combined WHERE parname=?''', (j,))
#     vals = cur.fetchall()
#     for i in vals:
#         (name, val, statErr, systErr) = i
#         err = np.sqrt(np.square(statErr)+np.square(systErr))
#         print(str(name).split('_')[0], '\t', str(name), '\t', str(val).replace('.',','),
#               '\t', str(statErr).replace('.',','), '\t', str(systErr).replace('.',','))
#              #'\t', str(err).replace('.',','))
#         # if j == 'delta_r_square':
#         #     print(name, '\t', round(val, 4), '('+str(round(statErr*10000))+')', '['+str(round(systErr*10000))+']' )
#         # else:
#         #     print(name, '\t', round(val, 2), '('+str(round(statErr*100))+')', '['+str(round(systErr*100))+']' )
#     con.close()