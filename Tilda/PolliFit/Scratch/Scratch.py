'''
Created on 31.03.2014

@author: gorges
'''

from Tilda.PolliFit import Tools

db = 'E:/Workspace/PolliFit/test/Project/CaD2_plain.sqlite'
file = open('E:/aknoerters/Projekte/TRIGA/Measurements and Analysis_Christian/Calcium43/43CaQI/Simuliert/Ca43QI_0deg.sp', 'w')
# file2 = open('E:/aknoerters/Projekte/TRIGA/Measurements and Analysis_Christian/Calcium43/43CaQI/Simuliert/Ca43QI_pi_2.sp', 'w')
# file3 = open('E:/aknoerters/Projekte/TRIGA/Measurements and Analysis_Christian/Calcium43/43CaQI/Simuliert/Ca43QI_magic.sp', 'w')
#
data = Tools.isoPlot(db, '40_Ca', linevar='_VoigtAsy', as_freq=False, laserfreq=761316951.2418399, col=0)
# data2 = Tools.isoPlot(db, '43_CaPi', linevar='_QI', as_freq=False, laserfreq=761316951.2418399, col=0)
# data3 = Tools.isoPlot(db, '43_CaMagic', linevar='_QI', as_freq=False, laserfreq=761316951.2418399, col=0)
for i, j in enumerate(data[0]):
    file.write(str(str(j) + str('\t') + str(data[1][i]) + str('\n')))
# for i, j in enumerate(data2[0]):
#     file2.write(str(str(j) + str('\t') + str(data2[1][i]) + str('\n')))
# for i, j in enumerate(data3[0]):
#     file3.write(str(str(j) + str('\t') + str(data3[1][i]) + str('\n')))

# run = ['Run0', 'Run3', 'Run4', 'Run5']
# for i in run:
#     # BatchFit.batchFit(['Ca43QI_pi_2.sp', 'Ca43QI_0deg.sp', 'Ca43QI_magic.sp'], db, run=i)
#     BatchFit.batchFit(['Ca_a_166.mcp', 'Ca_a_210.mcp', 'Ca_a_211.mcp'], db, run=i)
# a = Analyzer.combineRes('43_CaPi', 'Al', 'Run2', db)
# c = Analyzer.combineRes('43_CaPi', 'center', 'Run2', db)
# au = Analyzer.combineRes('43_CaPi', 'Au', 'Run2', db)
# bu = Analyzer.combineRes('43_CaPi', 'Bu', 'Run2', db)
#
# piDeg = str(str(np.round(c[0], 2)) + '\t' + str(np.round(c[1], 2)) + '\t' + str(np.round(au[0], 2)) +
#               '\t' + str(np.round(au[1], 2)) + '\t' + str(np.round(a[0], 2)) + '\t' + str(np.round(a[1], 2)) +
#               '\t' +str(np.round(bu[0], 2)) + '\t' + str(np.round(bu[1], 2)))
#
# a = Analyzer.combineRes('43_Ca0', 'Al', 'Run2', db)
# c = Analyzer.combineRes('43_Ca0', 'center', 'Run2', db)
# au = Analyzer.combineRes('43_Ca0', 'Au', 'Run2', db)
# bu = Analyzer.combineRes('43_Ca0', 'Bu', 'Run2', db)
#
# zeroDeg = str(str(np.round(c[0], 2)) + '\t' + str(np.round(c[1], 2)) + '\t' + str(np.round(au[0], 2)) +
#               '\t' + str(np.round(au[1], 2)) + '\t' + str(np.round(a[0], 2)) + '\t' + str(np.round(a[1], 2)) +
#               '\t' +str(np.round(bu[0], 2)) + '\t' + str(np.round(bu[1], 2)))
#
# k = []
# for i in run:
#     a = Analyzer.combineRes('43_Ca', 'Al', i, db)
#     c = Analyzer.combineRes('43_Ca', 'center', i, db)
#     au = Analyzer.combineRes('43_Ca', 'Au', i, db)
#     bu = Analyzer.combineRes('43_Ca', 'Bu', i, db)
#     k.append(str(str(np.round(c[0], 2)) + '\t' + str(np.round(c[1], 2)) + '\t' + str(np.round(au[0], 2)) +
#               '\t' + str(np.round(au[1], 2)) + '\t' + str(np.round(a[0], 2)) + '\t' + str(np.round(a[1], 2)) +
#               '\t' + str(np.round(bu[0], 2)) + '\t' + str(np.round(bu[1], 2))))
#
# print(zeroDeg)
# print(piDeg)
# for i in k:
#     print(i)

# print(np.round(c[0], 2), '(', np.round(c[1], 2), ') \t', np.round(au[0], 2), '(', np.round(au[1], 2), ') \t',
#       np.round(a[0], 2), '(', np.round(a[1], 2), ') \t', np.round(bu[0], 2), '(', np.round(bu[1], 2), ')')

