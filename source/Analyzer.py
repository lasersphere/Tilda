'''
Created on 21.05.2014

@author: hammen
'''

import sqlite3

import numpy as np

    
def extract(iso, par, run, st, db, fileList = ''):
    ''''''
    con = sqlite3.connect(db)
    cur = con.cursor()
    
    cur.execute('''SELECT file, pars FROM Results WHERE iso = ? AND run = ? AND sctr = ?''', (iso, run, repr(st)))
    fits = cur.fetchall()
    
    if fileList:
        fits = [f for f in fits if f[0] in fileList]
        
    fitres = [eval(f[1]) for f in fits]
    files = [f[0] for f in fits]
    vals = [f[par][0] for f in fitres]
    errs = [f[par][1] for f in fitres]
    
    con.close()
    return (files, vals, errs)
    
    
def weightedAverage(vals, errs):
    '''Return (weighted average, propagated error, rChi^2'''
    weights = 1 / np.square(errs)
    average = sum(vals * weights) / sum(weights)
    errorprop = 1 / sum(weights)
    if(len(vals) == 1):
        rChi = 0
    else:
        rChi = 1 / (len(vals) - 1) * sum(np.square(vals - average) * weights)
    
    return (average, errorprop, rChi)


def combineRes(iso, par, run, st, db):
    if par == 'shift':
        combineShift(iso, run, st, db)
        return
    
    print('Open DB', db)
    
    con = sqlite3.connect(db)
    cur = con.cursor()
    
    cur.execute('''SELECT (config, statErrForm, systErrForm) FROM Combined WHERE iso = ? AND par = ? AND run = ? AND sctr = ?''', (iso, par, run, repr(st)))
    (config, statErrForm, systErrForm) = cur.fetchall()[0]
    
    print('Combining', iso, par)
    files, vals, errs = extract(iso, par, run, st, db, config)
    
    val, err, rChi = weightedAverage(vals, errs)
    statErr = eval(statErrForm)
    systErr = eval(systErrForm)
    
    print('Combined to', iso, par, '=')
    print(str(val) + '(' + str(statErr) + ')[' + str(systErr) + ']')
    
    cur.execute('''UPDATE Combined SET val = ?, statErr = ?, systErr = ?, rChi = ?
        WHERE iso = ? AND par = ? AND run = ? AND scaltr = ?''', (val, statErr, systErr, rChi, iso, par, run, st))

    con.commit()
    con.close()
    

def combineShift(iso, run, st, db):
    print('Open DB', db)
    
    con = sqlite3.connect(db)
    cur = con.cursor()
    
    cur.execute('''SELECT (config, statErrForm, systErrForm) FROM Combined WHERE iso = ? AND par = ? AND run = ? AND sctr = ?''', (iso, 'shift', run, repr(st)))
    (config, statErrForm, systErrForm) = cur.fetchall()[0]
    
    print('Combining', iso, 'shift')
    
    #each block is used to measure the isotope shift once
    for block in config:
        pass
    

    
    
def applyChi(err, *rChi):
    '''Increases error by sqrt(rChi^2) if necessary. Works for several rChi as well'''
    return err * np.max(1, np.max(np.sqrt(rChi)))

def gaussProp(*args):
    '''Calculate sqrt of squared sum of args, as in gaussian error propagation'''
    return np.sqrt(sum(x**2 for x in args))