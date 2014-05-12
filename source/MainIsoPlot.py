'''
Created on 12.05.2014

@author: hammen
'''

from matplotlib import pyplot as plt

from DBIsotope import DBIsotope
from Spectra.FullSpec import FullSpec


if __name__ == '__main__':
    niso = '2_Mi-D0'
    ndb = 'iso.sqlite'
    
    iso = DBIsotope(niso, ndb)
    
    spec =  FullSpec(iso)
    
    data = spec.toPlot(spec.getPars())
    
    plt.plot(data[0], data[1], 'k-')
    plt.show()