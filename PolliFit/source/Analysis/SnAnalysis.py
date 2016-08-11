'''
Created on 31.03.2014

@author: gorges
'''
import sqlite3

import Analyzer
import BatchFit
import Physics
import Tools

db = 'V:/Projekte/COLLAPS/Sn/Measurement_and_Analysis_Christian/Sn.sqlite'

'''preparing a list of isotopes'''
# isoL = ['112_Sn']
# for i in range(114,121, 1):
#     isoL.append(str(str(i)+'_Sn'))
# isoL.append('122_Sn')
isoL = []
for i in range(109,134, 1):
    isoL.append(str(str(i)+'_Sn'))
# freq = 662305065
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

'''Crawling'''
# Tools.crawl(db)
redChisqs = [[]]
divratios = []
accratios = []
l = 0
print(Physics.diffDoppler(662378005, 40000, 124))
for k in range(10006, 10007, 1):
    acc = k
    for i in range(10017,10018, 1):
        freq = 662378005 +(10000-acc)*27.5
        i = i/10
        con = sqlite3.connect(db)
        cur = con.cursor()
        cur.execute('''UPDATE Files SET laserFreq=662929863.205568,
          voltDivRatio="{'accVolt':10001, 'offset':1000}",
          lineMult=0.050425, lineOffset=0.00015''')
        divratio = str(''' voltDivRatio="{'accVolt':''' + str(k) + ''', 'offset':''' + str(i) + '''}" ''')
        cur.execute('''UPDATE Files SET ''' + divratio)

        cur.execute('''UPDATE Lines SET frequency='''+str(freq))

        con.commit()
        con.close()
        divratios.append(i)
        accratios.append(k)

        '''Fitting the Kepco-Scans!'''
        # for i in range(0,2):
        #     run = 'ZKepRun' + str(i)
        #     BatchFit.batchFit(Tools.fileList(db,'Kepco'), db, run)
        #     Analyzer.combineRes('Kepco', 'm', run, db, show_plot=False)
        #     Analyzer.combineRes('Kepco', 'b', run, db, show_plot=False)

        '''Fitting the spectra with Voigt-Fits!'''
        isotopeShifts = []
        isotopeShiftErrs = []
        BatchFit.batchFit(Tools.fileList(db,'124_Sn'), db,'Run0')
        Analyzer.combineRes('124_Sn', 'sigma', 'Run0', db)
        Analyzer.combineRes('124_Sn', 'center', 'Run0', db)
        for j in isoL:
            '''if there is an isomer at the isotope, we need Run1:'''
            if j == '125_Sn' or j == '127_Sn' or j == '129_Sn' or j == '130_Sn' or j == '131_Sn':
                run = 'Run1'
            else:
                run = 'Run0'

            if j == '109_Sn' or j == '110_Sn' or j == '111_Sn' or j == '113_Sn' or j == '121_Sn' or j == '123_Sn' or j == '129_Sn' or j == '131_Sn':
                isotopeShifts.append('---')
                isotopeShiftErrs.append('---')
            elif j == '117_Sn':
                BatchFit.batchFit(['117Sn_no_protonTrigger_Run141.mcp','117Sn_no_protonTrigger_Run142.mcp',
                                   '117Sn_no_protonTrigger_Run179.mcp'], db, 'Run1')
                BatchFit.batchFit(['117Sn_no_protonTrigger_Run033.mcp','117Sn_no_protonTrigger_Run034.mcp'], db, run)
                shift = Analyzer.combineShift(j, run, db)
                isotopeShifts.append(shift[2])
                statErr = float(shift[3])*2
                # isotopeShiftErrs.append(np.sqrt(np.square(statErr)+np.square(float(shift[4]))))
                isotopeShiftErrs.append(statErr)
            elif j == '119_Sn':
                BatchFit.batchFit(['119Sn_no_protonTrigger_Run137.mcp','119Sn_no_protonTrigger_Run174.mcp'], db, 'Run1')
                BatchFit.batchFit(['119Sn_no_protonTrigger_Run028.mcp','119Sn_no_protonTrigger_Run029.mcp',
                                   '119Sn_no_protonTrigger_Run030.mcp','119Sn_no_protonTrigger_Run055.mcp',
                                   '119Sn_no_protonTrigger_Run153.mcp'], db, run)
                shift = Analyzer.combineShift(j, run, db)
                isotopeShifts.append(shift[2])
                statErr = float(shift[3])*2
                # isotopeShiftErrs.append(np.sqrt(np.square(statErr)+np.square(float(shift[4]))))
                isotopeShiftErrs.append(statErr)
            else:
                BatchFit.batchFit(Tools.fileList(db,j), db,run)
                '''Calculate the isotope shift to 124_Sn'''
                shift = Analyzer.combineShift(j, run, db)
                isotopeShifts.append(shift[2])
                statErr = shift[3]*2
                # isotopeShiftErrs.append(np.sqrt(np.square(float(statErr))+np.square(float(shift[4]))))
                isotopeShiftErrs.append(float(statErr))
                '''Mean of center, sigma and gamma for the isotopes'''
                # Analyzer.combineRes(j, 'center',run, db, show_plot=False)
                Analyzer.combineRes(j, 'sigma',run, db, show_plot=False)
                '''comparison if intensities'''
                # Ints0 = Analyzer.combineRes(j, 'Int0',run, db)
                # Ints1 = Analyzer.combineRes(j, 'Int1',run, db)
                # ints = []
                # for i,j in enumerate(Ints0[3][1]):
                #     ints.append(float(Ints1[3][1][i]/j))
                # print(ints)

        '''calculating red. Chi^2'''
    #     litvals = [-1380.3, -1115.7, -1044.3, -842.6, -759.1, -588.4, -521.4, -360.3, -162.8]
    #     literrs = [8.9, 8.3, 8.3, 8, 8.1, 8.3, 8.3, 8.9, 10]
    #     chisq = 0
    #     for i in range(0,len(isoL)):
    #         print(isoL[i])
    #         print(isotopeShifts[i])
    #         print(litvals[i])
    #         chisq += np.square((isotopeShifts[i]-litvals[i])/(isotopeShiftErrs[i]))
    #         print(chisq)
    #     redChisqs[l].append(float(chisq/9))
    #     print(divratios)
    #     print('red Chi^2:', redChisqs[l])
    # l+=1
    # redChisqs.append([])

'''producing an output that can be used with origin or excel:'''
isoL = str(isoL)
isotopeShifts  = str(isotopeShifts)
isotopeShiftErrs  = str(isotopeShiftErrs)
# isoL = str(accratios)
# isotopeShifts = str(divratios)

strisoL = ''
for i in isoL:
    if i != "'" and i != "[" and i != "S" and i != "n" and i != "_" and i != "]":
        strisoL += i
strisoL = strisoL.replace(",", "\t")
strisoL = strisoL.replace(".", ",")

strIsotopeShifts = ''
for i in isotopeShifts:
    if i != "'" and i != "[" and i != "]":
        strIsotopeShifts += str(i)
strIsotopeShifts = strIsotopeShifts.replace(",", "\t")
strIsotopeShifts = strIsotopeShifts.replace(".", ",")
print(strisoL)
print(strIsotopeShifts)

# for k in redChisqs:
#     isotopeShiftErrs = str(k)
strIsotopeShiftErrs = ''
for i in isotopeShiftErrs:
    if i != "'" and i != "[" and i != "]":
        strIsotopeShiftErrs += i
strIsotopeShiftErrs = strIsotopeShiftErrs.replace(",", "\t")
strIsotopeShiftErrs = strIsotopeShiftErrs.replace(".", ",")
print(strIsotopeShiftErrs)
