'''
Created on 20.01.2017

@author: schmid
'''

import matplotlib.pyplot as mpl
import sys
sys.path.append('C:/Workspace/PolliFit/source')
from Measurement.BeaImporter import BeaImporter



path = 'C:/Users/Laura/Desktop/Bea Bachelorarbeit/Daten/SimData03.txt'
bi = BeaImporter(path,14,380680000,True)

print(bi.x[0])

mpl.plot(bi.x[0],bi.cts[0][0])
#mpl.plot(bi.cts[0][0])
mpl.show()
