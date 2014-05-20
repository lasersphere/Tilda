'''
Created on 12.05.2014

@author: hammen
'''

from matplotlib import pyplot as plt

from DBIsotope import DBIsotope
from Spectra.FullSpec import FullSpec
import Physics


if __name__ == '__main__':
    niso = '40_Ca'
    nline = "Ca-D1"
    ndb = 'V:/Projekte/A2-MAINZ-EXP/TRIGA/Measurements and Analysis_Christian/Calcium Isotopieverschiebung/397nm_14_05_13/calciumD1.sqlite'
    iso = DBIsotope(niso, nline, ndb)
    
    spec =  FullSpec(iso)
    
    print(spec.getPars())
    
    data = spec.toPlotE(Physics.freqFromWavenumber(12586.300*2), False, spec.getPars())
    
    plt.plot(data[0], data[1], 'k-')
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    plt.show()