'''
Created on 29.04.2014

@author: hammen
'''

import matplotlib.pyplot as plt
import numpy as np
    
    
def plot(*args):
    for a in args:
        plt.plot(a[0], a[1])
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    plt.ylabel('Intensity / a.u.')
    plt.xlabel('Frequency / MHz')
    
    
def plotFit(fit):    
    data = fit.meas.getArithSpec(*fit.st)
    plotdat = fit.spec.toPlotE(fit.meas.laserFreq, fit.meas.col, fit.par)


    fig = plt.figure(1, (8, 8))
    fig.patch.set_facecolor('white')

    ax1 = plt.axes([0.15, 0.35, 0.8, 0.6])
    plt.errorbar(data[0], data[1], yerr = data[2], fmt = 'k.')
    plt.plot(plotdat[0], plotdat[1], 'r-')
    ax1.get_xaxis().get_major_formatter().set_useOffset(False)

    ax2 = plt.axes([0.15, 0.1, 0.8, 0.2])
    plt.errorbar(data[0], fit.calcRes(), yerr = data[2], fmt = 'k.')
    ax2.get_xaxis().get_major_formatter().set_useOffset(False)
    plt.ylabel('Intensity / a.u.')
    plt.xlabel('Ion kinetic energy / eV')
    

def plotAverage(lin, lout, aver, err):
    
    


def show():
    plt.show()
    
def ion():
    plt.ion()
    
def save(file):
    plt.savefig(file)
    
def clear():
    plt.clf()