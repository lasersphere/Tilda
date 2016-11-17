'''
Created on 20.09.2016

@author: gorges
'''
import sqlite3

import numpy as np

import Analyzer
import BatchFit
import Tools

db = 'V:/Projekte/COLLAPS/ROC/optical/CaD2_optical.sqlite'

'''Crawling'''
#Tools.crawl(db)
#print(Physics.freqFromWavenumber(12722.2986*2))

'''Fitting the Kepco-Scans!'''
# BatchFit.batchFit(Tools.fileList(db,'Kepco'), db, 'Run2')
# Analyzer.combineRes('Kepco', 'm', 'Run2', db, False)
# Analyzer.combineRes('Kepco', 'b', 'Run2', db, False)

'''Fitting the spectra with Voigt-Fits!'''
shift40 = []
shift42 = []
shift44 = []

isoL=['40_Ca','42_Ca','44_Ca']
chiSqs = [[]]
chiSqs2 = [[]]
divratios = []
accratios = []
l = 0
# print(Physics.diffDoppler(662378005, 40000, 124))
run = 'Run5'
freq = 761906723.3
for k in range(99444, 99445, 1):
    k=k/100
    acc = k
    for i in range(99517, 99518, 1):
        #i = i+ind
        freq = 761906723.3+(1000-acc)*432
        i = i/100
        con = sqlite3.connect(db)
        cur = con.cursor()
        cur.execute('''UPDATE Lines SET frequency='''+str(freq)+''' WHERE refRun="Run5"''')

        divratio = str(''' voltDivRatio="{'accVolt':''' + str(k) + ''', 'offset':''' + str(i) + '''}" ''')
        cur.execute('''UPDATE Files SET ''' + divratio)
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
        BatchFit.batchFit(Tools.fileList(db,'48_Ca'), db,run)

        isotopeShifts = []
        isotopeShiftErrs = []
        for j in isoL:
                BatchFit.batchFit(Tools.fileList(db,j), db,run)
                '''Calculate the isotope shift to 48_Ca'''
                shift = Analyzer.combineShift(j, run, db)
                isotopeShifts.append(shift[2])
                isotopeShiftErrs.append(np.sqrt(np.square(10*shift[3])))

        '''calculating red. Chi^2'''
        litvals = [-1707.945, -1282.013, -857.714]
        chisq = 0
        print(isotopeShifts)
        print(isotopeShiftErrs)
        for index in range(0, len(isoL)):
            chisq += np.square((isotopeShifts[index]-litvals[index])/(np.sqrt(np.square(isotopeShiftErrs[index]))))
        chiSqs[l].append(float(chisq))
        print('red Chi^2:', chiSqs[l])
        print(k)
        print(i)
    print(k)
    l+=1
    chiSqs.append([])
strA = str(accratios).replace(',','\t')
strA = strA.replace('.', ',')
print(strA[1:-1])
strA = str(divratios).replace(',','\t')
strA = strA.replace('.', ',')
print(strA[1:-1])
for i in chiSqs:
    strI = str(i).replace(',','\t')
    strI = strI.replace('.', ',')
    print(strI[1:-1])