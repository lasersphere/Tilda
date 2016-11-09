'''
Created on 11.07.2016

@author: gorges
'''

from KingFitter import KingFitter

db = 'V:/Projekte/COLLAPS/Sn/Measurement_and_Analysis_Christian/Sn.sqlite'

'''preparing a list of isotopes'''
# isoL = ['109_Sn','112_Sn']
# for i in range(114,121, 1):
#     isoL.append(str(str(i)+'_Sn'))
# isoL.append('122_Sn')
isoL = ['109_Sn']
for i in range(112,135, 1):
    isoL.append(str(str(i)+'_Sn'))
freq = 662305065
# isoL = ['112_Sn','114_Sn','115_Sn','116_Sn','118_Sn','119_Sn','120_Sn','122_Sn','125_Sn','126_Sn','128_Sn','131_Sn','132_Sn','133_Sn','134_Sn']

'''Plotting spectra'''
# Tools.centerPlot(db,isoL)

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
# for k in range(99971, 99972, 1):
#     k=k/10
#     acc = k
#
#     for i in range(100085, 100086, 1):
#         # freq = 662378005 +(10000-acc)*27.5
#         # #i = i+ind
#         # i = i/100
#         # con = sqlite3.connect(db)
#         # cur = con.cursor()
#         # cur.execute('''UPDATE Files SET laserFreq=662929863.205568,
#         #   voltDivRatio="{'accVolt':9997.1, 'offset':1000.85}",
#         #   lineMult=0.050425, lineOffset=0.00015''')
#         # divratio = str(''' voltDivRatio="{'accVolt':''' + str(k) + ''', 'offset':''' + str(i) + '''}" ''')
#         # cur.execute('''UPDATE Files SET ''' + divratio)
#         #
#         # cur.execute('''UPDATE Lines SET frequency='''+str(freq))
#         #
#         # con.commit()
#         # con.close()
#         # divratios.append(i)
#         # accratios.append(k)
#
#         '''Fitting the Kepco-Scans!'''
#         # for i in range(0,2):
#         #     run = 'ZKepRun' + str(i)
#         #     BatchFit.batchFit(Tools.fileList(db,'Kepco'), db, run)
#         #     Analyzer.combineRes('Kepco', 'm', run, db, show_plot=False)
#         #     Analyzer.combineRes('Kepco', 'b', run, db, show_plot=False)
#
#         '''Fitting the spectra with Voigt-Fits!'''
#         BatchFit.batchFit(Tools.fileList(db,'124_Sn'), db,'Run0')
#         # BatchFit.batchFit(['124Sn_no_protonTrigger_Run018.mcp','124Sn_no_protonTrigger_Run021.mcp','124Sn_no_protonTrigger_Run024.mcp',
#         #                    '124Sn_no_protonTrigger_Run025.mcp','124Sn_no_protonTrigger_Run031.mcp','124Sn_no_protonTrigger_Run032.mcp',
#         #                    '124Sn_no_protonTrigger_Run037.mcp','124Sn_no_protonTrigger_Run038.mcp','124Sn_no_protonTrigger_Run043.mcp',
#         #                    '124Sn_no_protonTrigger_Run054.mcp','124Sn_no_protonTrigger_Run056.mcp','124Sn_no_protonTrigger_Run057.mcp',
#         #                    '124Sn_no_protonTrigger_Run061.mcp','124Sn_no_protonTrigger_Run138.mcp','124Sn_no_protonTrigger_Run139.mcp',
#         #                    '124Sn_no_protonTrigger_Run143.mcp','124Sn_no_protonTrigger_Run152.mcp','124Sn_no_protonTrigger_Run154.mcp',
#         #                    '124Sn_no_protonTrigger_Run175.mcp','124Sn_no_protonTrigger_Run176.mcp','124Sn_no_protonTrigger_Run182.mcp',
#         #                    '124Sn_no_protonTrigger_Run195.mcp','124Sn_no_protonTrigger_Run204.mcp',], db,'Run0')
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
#
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
'''performing a King fit analysis'''
# litvals = {'112_Sn':[-0.748025649,.0077],
#             '114_Sn':[-0.601624554,.0077],
#            '116_Sn':[ -0.464108311,.0077],
#             '117_Sn':[-0.422258642,.0075],
#            '118_Sn':[-0.327818629,.0077],
#            '119_Sn':[-0.303343067,.0075],
#             '120_Sn':[-0.202198458,.0080],
#             '122_Sn':[-0.093007073,.0077]}#Fricke charge radii

# litvals = {'112_Sn':[-1659.44,0.21],'113_Sn':[-1520.45,2.30],'114_Sn':[-1341.83, 0.21],'115_Sn':[-1246.07,0.19],
#             '116_Sn':[-1017.19,0.21],'117_Sn':[-912.58,0.19],'118_Sn':[-711.39,0.21],'119_Sn':[-620.74,0.19],
#            '120_Sn':[-441.15,0.15],'121_Sn':[-350.35,2.70],'122_Sn':[-205.8,0.21],'123_Sn':[-143.95,2.01],
#            '125_Sn':[63.15,7.5],'117_Sn_m':[-924.45,1.81],'121_Sn_m':[-369.15,1.71]}#Anselment isotope shift

king = KingFitter(db, showing=True)
run = -1
isotopes = ['122_Sn','132_Sn','133_Sn']
king.kingFit(alpha=-849,findBestAlpha=False, run=run)
king.calcChargeRadii(isotopes=isotopes,run=run)
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
#         # print(str(name).split('_')[0], '\t', str(name), '\t', str(val).replace('.',','),
#         #       '\t', str(statErr).replace('.',','), '\t', str(systErr).replace('.',','))
#         #      '\t', str(err).replace('.',','))
#         if j == 'delta_r_square':
#             print(name, '\t', round(val, 4), '('+str(round(statErr*10000))+')', '['+str(round(systErr*10000))+']' )
#         else:
#             print(name, '\t', round(val, 2), '('+str(round(statErr*100))+')', '['+str(round(systErr*100))+']' )
#     con.close()