'''
Created on 12.09.2017

@author: gorges
'''

from Tilda.PolliFit import Tools

db = 'C:/Workspace/PolliFit/test/Project/Cadmium.sqlite'
#
file = open('C:/Workspace/PolliFit/test/Project/Data/Cd128.dat', 'w')
isotope_list = ['100_Cd', '101_Cd', '102_Cd', '103_Cd', '104_Cd', '105_Cd', '106_Cd', '107_Cd', '108_Cd', '109_Cd', '110_Cd', '111_Cd', '112_Cd', '113_Cd', '114_Cd', '115_Cd', '116_Cd', '121_Cd', '123_Cd', '124_Cd', '125_Cd', '126_Cd', '127_Cd', '128_Cd', '129_Cd', '130_Cd']
isomer_list = ['111_Cd', '119_Cd', '127_Cd']
for i in isotope_list:
    data = Tools.isoPlot(db, i, saving=True, show=False, prec=5000)
    file.write(str('\n') + i + str('\n \n'))
    for j, k in enumerate(data[0]):
        file.write(str(str(k) + str('\t') + str(data[1][j]) + str('\n')))
for i in isomer_list:
    data = Tools.isoPlot(db, i, saving=True, isom_name=str(i) + str('_m'), prec=5000)

#Tools.centerPlot(db, isotope_list)


