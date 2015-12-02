'''
Created on 29.04.2014

@author: hammen
'''

from matplotlib.dates import DateFormatter
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os

import Analyzer


def plot(*args):
    for a in args:
        plt.plot(a[0], a[1])
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    plt.ylabel('Intensity [cts]')
    plt.xlabel('Frequency [MHz]')
    
    
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
    

def plotAverage(date, cts, errs, avg, stat_err, syst_err, forms=('k.', 'r')):
    # avg, stat_err, sys_err = Analyzer.combineRes(iso, par, run, db, print_extracted=False)
    # val, errs, date = Analyzer.extract(iso, par, run, db, prin=False)
    date = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in date]
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=25)
    ax = plt.gca()
    xfmt = DateFormatter('%Y-%m-%d %H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)
    plt.errorbar(date, cts, yerr=errs, fmt=forms[0])
    err_p = avg+abs(stat_err)+abs(syst_err)
    err_m = avg-abs(stat_err)-abs(syst_err)
    err_p_l = np.full((2,), err_p)
    err_m_l = np.full((2,), err_m)
    x = (sorted(date)[0], sorted(date)[-1])
    y = (avg, avg)
    plt.plot(x, y, forms[1])
    plt.fill_between(x, err_p_l, err_m_l, alpha=0.5)


def show(block=True):
    plt.show(block=block)


def ion():
    plt.ion()


def save(file):
    plt.savefig(file, dpi=100)


def clear():
    plt.clf()


def draw():
    plt.draw()


def pause(time):
    plt.pause(time)


def plt_axes(axes, title, plotlist):
    axes.clear()
    axes.plot(*plotlist)
    axes.set_ylabel(title)


def get_current_axes():
    return plt.gca()


def get_current_figure():
    return plt.gcf()