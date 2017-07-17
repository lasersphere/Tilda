'''
Created on 04.08.2015

@author: gorges
'''

import Tools, Analyzer, Physics, BatchFit, sqlite3
import numpy as np

db = 'E:/aknoerters/Projekte/TRIGA/Measurements and Analysis_Christian/Calcium43/43CaQI/CaD2_plain.sqlite'

db = 'E:/Workspace/PolliFit/test/Project/Ca_plain.sqlite'
'''Plotting the Isotopes'''
# isoL = ['42_Ca', '44_Ca', '48_Ca']
run = ['Run0', 'Run2', 'Run3', 'Run4', 'Run5']

# for i in range(40,48,2):
#     iso = str(str(i) + '_Ca')
#     isoL.append(iso)
# print(Physics.freqFromWavenumber(2*12722.2982))
# Tools.isoPlot(db, '40_Ca', linevar='QI', laserfreq=761316951.2418399, as_freq=False)
Tools.isoPlot(db, '40_Ca', linevar='_fano')
#Tools.centerPlot(db, ['48_Ca'])

'''Crawling'''
# #Tools.crawl(db, 'DataKepco')
# Tools.crawl(db, 'DataD1')
# Tools.crawl(db)

'''Fitting the Kepco-Scans!'''
# BatchFit.batchFit(Tools.fileList(db,'Kepco'), db, 'Run0')
# Analyzer.combineRes('Kepco', 'm', 'Run0', db, False)
# Analyzer.combineRes('Kepco', 'b', 'Run0', db, False)

'''Fitting the spectra with Voigt-Fits!'''
# centers = []
# centerErrs = []
# als = []
# alsErrs = []
# aus = []
# ausErrs = []
# bus = []
# busErrs = []
# for i in run:
#     # BatchFit.batchFit(['Ca_a_166.mcp', 'Ca_a_210.mcp', 'Ca_a_211.mcp'], db, i)
# #         BatchFit.batchFit(Tools.fileList(db,'44_Ca'), db,run)
#
#     Analyzer.combineRes('43_Ca', 'center', i, db)
#     Analyzer.combineRes('43_Ca', 'Al', i, db)
#     Analyzer.combineRes('43_Ca', 'Au', i, db)
#     Analyzer.combineRes('43_Ca', 'Bu', i, db)
#
#     con = sqlite3.connect(db)
#     cur = con.cursor()
#     cur.execute('''SELECT val, statErr FROM Combined WHERE parname=? AND run = ?''', ('center',i))
#     (k, l) = cur.fetchall()[0]
#     centers.append(np.round(k, 2))
#     centerErrs.append(np.round(l, 2))
#     cur.execute('''SELECT val, statErr FROM Combined WHERE parname=? AND run = ?''', ('Al',i))
#     (k, l) = cur.fetchall()[0]
#     als.append(np.round(k, 2))
#     alsErrs.append(np.round(l, 2))
#     cur.execute('''SELECT val, statErr FROM Combined WHERE parname=? AND run = ?''', ('Au',i))
#     (k, l) = cur.fetchall()[0]
#     aus.append(np.round(k, 2))
#     ausErrs.append(np.round(l, 2))
#     cur.execute('''SELECT val, statErr FROM Combined WHERE parname=? AND run = ?''', ('Bu',i))
#     (k, l) = cur.fetchall()[0]
#     bus.append(np.round(k, 2))
#     busErrs.append(np.round(l, 2))
#     con.close()
#
# for i, j in enumerate(run):
#     print(j, '\t', centers[i], '(', centerErrs[i], ') \t ', aus[i], '(', ausErrs[i], ') \t ', als[i], '(', alsErrs[i],
#           ') \t ', bus[i], '(', busErrs[i], ')')