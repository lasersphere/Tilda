'''
Created on 12.09.2017

@author: gorges
'''

import BatchFit
import Tools
import Analyzer
import numpy as np
import Physics
import copy

db = 'E:/Workspace/PolliFit/test/Project/Cadmium.sqlite'
#
file = open('E:/Workspace/PolliFit/test/Project/Data/Cd.sp', 'w')
isotope_list = ['111_Cd', '111_Cd_m', '114_Cd', '116_Cd', '118_Cd', '119_Cd', '119_Cd_m', '127_Cd',  '127_Cd_m']
isomer_list = ['111_Cd', '119_Cd', '127_Cd']
for i in isotope_list:
    data = Tools.isoPlot(db, i, saving=True, show=False, prec=5000)
    file.write(str('\n') + i + str('\n \n'))
    for j, k in enumerate(data[0]):
        file.write(str(str(k) + str('\t') + str(data[1][j]) + str('\n')))
for i in isomer_list:
    data = Tools.isoPlot(db, i, saving=True, isom_name=str(i)+str('_m'), prec=5000)

#Tools.centerPlot(db, isotope_list)


