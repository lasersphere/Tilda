'''
Created on 18.05.2014

@author: hammen
'''

import matplotlib.pyplot as plt
import numpy as np
from DBIsotope import DBIsotope
import Physics


def centerPlot(isoL, line, db):
    
    isos = [DBIsotope(iso, line, db) for iso in isoL]
    
    width = 1e6
    res = 100
    fx = np.linspace(isos[0].freq - width, isos[0].freq + width, res)
    wnx = Physics.wavenumber(fx)
    
    y = np.zeros((len(isos), len(fx)))
    for i, iso in enumerate(isos):
        for j, x in enumerate(fx):
            v = Physics.invRelDoppler(x, iso.freq + iso.center)
            y[i][j] = (iso.mass * Physics.u * v**2)/2 / Physics.qe
    
    
    fig = plt.figure(1, (8, 8))
    fig.patch.set_facecolor('white')
    
    for i in y:
        plt.plot(wnx, i, '-')
    
    plt.xlabel("Laser wavenumber / cm^-1")
    plt.ylabel("Ion energy on resonance / eV")
    plt.axvline(Physics.wavenumber(isos[0].freq), 0, 20000)
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    plt.show()


if __name__ == '__main__':
    centerPlot(['40_Ca', '42_Ca', '44_Ca', '48_Ca'], 'Ca-D1', "calciumD1.sqlite")