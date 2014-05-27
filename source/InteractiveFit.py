'''
Created on 27.05.2014

@author: hammen

InteractiveFit is an interactive fitting wrapper that can be used to easily determine starting parameters.
Import the module in IPython and load() a file. The spectrum as well as the datapoints will be plotted.
printPars() is used to plot starting values, setPar() to change them, fit() to try and fit, reset() to
revert to the values before the last fit try.
'''
import sqlite3
import ast
import os

import MPLPlotter as plot

import Measurement.MeasLoad as MeasLoad
from Spectra.Straight import Straight
from Spectra.FullSpec import FullSpec
from DBIsotope import DBIsotope
from SPFitter import SPFitter

fitter = None

def load(file, db, run):
    global fitter
    global oldpar
    plot.ion()
    plot.clear()
    con = sqlite3.connect(db)
    cur = con.cursor()
    
    cur.execute('''SELECT filePath FROM Files WHERE file = ?''', (file,))
    
    try:
        path = os.path.join(os.path.dirname(db), cur.fetchall()[0][0])
    except:
        raise Exception(str(file) + " not found in DB")
    
    print('Loading file', path)
    
    cur.execute('''SELECT isoVar, lineVar, scaler, track FROM Runs WHERE run = ?''', (run,))
    var = cur.fetchall()[0]
    st = (ast.literal_eval(var[2]), ast.literal_eval(var[3]))
    
    meas = MeasLoad.load(path, db)
    if meas.type == 'Kepco':
        spec = Straight()
    else:
        iso = DBIsotope(db, meas.type, var[0], var[1])
        spec = FullSpec(iso)
        
    fitter = SPFitter(spec, meas, st)
    plot.plotFit(fitter)
    plot.show()
    
    
def printPars():
    print('Current parameters:')
    for n, p, f in zip(fitter.npar, fitter.par, fitter.fix):
        print(n + '\t' + str(p) + '\t' + str(f))
        
def fit():
    fitter.fit()
    plot.clear()
    plot.plotFit(fitter)
    plot.show()
    
def reset():
    fitter.reset()
    plot.clear()
    plot.plotFit(fitter)
    plot.show()
    
def setPar(name, par):
    fitter.setPar(name, par)
    plot.clear()
    plot.plotFit(fitter)
    plot.show()

def setPars(par):
    fitter.par = par
    plot.clear()
    plot.plotFit(fitter)
    plot.show()
    
