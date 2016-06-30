'''
Created on 31.03.2014

@author: hammen
'''

from Measurement.SimpleImporter import SimpleImporter
import MPLPlotter as plot

from DBIsotope import DBIsotope
from SPFitter import SPFitter
from Spectra.FullSpec import FullSpec

import os
import numpy as np

# path = "../test/cd_c_137data.txt"
# file = SimpleImporter(path)
# iso = DBIsotope('114_Mi', 'Mi-D0',  '../test/iso.sqlite')
# spec = FullSpec(iso)
#
# fit = SPFitter(spec, file, (0, -1))
#
# print(fit.spec.parAssign())
#
# fit.fit()
#
#
# plot.plotFit(fit)
# plot.show()
abc = []
cde = []
efg = []
for j in range(0,5):
    lists = []
    for i in range(0,3):
        lists.append(j)
    abc.append(lists)
for j in range(0,6,2):
    lists = []
    for i in range(0,2):
        lists.append(j)
    cde.append(lists)
for j in range(0,8,4):
    lists = []
    for i in range(0,2):
        lists.append(j)
    efg.append(lists)
cts = []
cts.append(abc)
cts.append(cde)
cts.append(efg)
print(cts)
list = ['0','1','2','3','4']
list2 = ['0','2','4']
list3 = ['0','4']
activePMTs = []
activePMTs.append(list)
activePMTs.append(list2)
activePMTs.append(list3)
for track in range(0, len(activePMTs)-1):
    diff = len(activePMTs)-track
    for multi in range(1,diff):
        activePMTcopy = activePMTs[track].copy()
        activePMTcopy2 = activePMTs[track+multi].copy()
        eraser = 0
        for scaler in range(0,len(activePMTs)):
            indices = [i for i, x in enumerate(activePMTcopy2) if activePMTcopy[scaler] == x]
            if indices == []:
                cts[track].pop(eraser)
                activePMTs[track].pop(eraser)
                eraser-=1
            eraser+=1
errs = cts.copy()
for i,j in enumerate(cts):
    errs[i] = np.sqrt(j)
print(cts, activePMTs)
print(errs)

#Test