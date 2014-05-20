'''
Created on 29.04.2014

@author: hammen
'''

import matplotlib.pyplot as plt
import numpy as np
    
    
def printSpec(self, spec, par):
    x = np.linspace(spec.leftEdge(), spec.rightEdge(), 10000)
    y = np.fromiter((spec.evaluate([m], par) for m in x), np.float32)

    plt.plot(x, y)
    plt.ylabel('Intensity / a.u.')
    plt.xlabel('Frequency / MHz')
        
    plt.draw()
    
    
def plotFit(fit):
    
    data = fit.meas.getSingleSpec(0, -1)
    plotdat = fit.spec.toPlotE(fit.meas.laserFreq, fit.meas.col, fit.par)


    fig = plt.figure(1, (8, 8))
    fig.patch.set_facecolor('white')

    ax1 = plt.axes([0.1, 0.35, 0.8, 0.6])
    plt.errorbar(data[0], data[1], yerr = data[2], fmt = 'k.')
    plt.plot(plotdat[0], plotdat[1], 'r-')
    ax1.get_xaxis().get_major_formatter().set_useOffset(False)

    ax2 = plt.axes([0.1, 0.05, 0.8, 0.25])
    plt.errorbar(data[0], fit.calcRes(), yerr = data[2], fmt = 'k.')
    ax2.get_xaxis().get_major_formatter().set_useOffset(False)
    plt.ylabel('Intensity / a.u.')
    plt.xlabel('Ion kinetic energy / eV')
    
    
def show():
    plt.show()
    
def save(file):
    plt.savefig(file)
    
def clear():
    plt.clf()