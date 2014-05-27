'''
Created on 21.05.2014

@author: hammen
'''

import sqlite3
import ast

import numpy as np

    
def extract(iso, par, run, st, db, fileList = ''):
    ''''''
    con = sqlite3.connect(db)
    cur = con.cursor()
    
    cur.execute('''SELECT file, pars FROM FitRes WHERE iso = ? AND run = ? AND sctr = ?''', (iso, run, repr(st)))
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
    print('Open DB', db)
    
    con = sqlite3.connect(db)
    cur = con.cursor()
    
    cur.execute('''SELECT config, statErrForm, systErrForm FROM Combined WHERE iso = ? AND parname = ? AND run = ? AND sctr = ?''', (iso, par, run, repr(st)))
    (config, statErrForm, systErrForm) = cur.fetchall()[0]
    config = ast.literal_eval(config)
    
    print('Combining', iso, par)
    files, vals, errs = extract(iso, par, run, st, db, config)
    
    for f, v, e in zip(files, vals, errs):
        print(f, '\t', v, '\t', e)
    
    val, err, rChi = weightedAverage(vals, errs)
    statErr = eval(statErrForm)
    systErr = eval(systErrForm)
    
    print('Statistical error formula:', statErrForm)
    print('Systematic error formula:', systErrForm)
    print('Combined to', iso, par, '=')
    print(str(val) + '(' + str(statErr) + ')[' + str(systErr) + ']')
    
    cur.execute('''UPDATE Combined SET val = ?, statErr = ?, systErr = ?, rChi = ?
        WHERE iso = ? AND parname = ? AND run = ? AND sctr = ?''', (val, statErr, systErr, rChi, iso, par, run, repr(st)))

    con.commit()
    con.close()
    

def combineShift(iso, run, st, db):
    '''[(['dataREF1.*','dataREF2',...],['dataINTERESTING1.*','dataINT2.*',...],['dataREF4.*',...]), ([...],[...],[...]), ...]'''
    print('Open DB', db)
    
    con = sqlite3.connect(db)
    cur = con.cursor()
    
    cur.execute('''SELECT config, statErrForm, systErrForm FROM Combined WHERE iso = ? AND par = ? AND run = ? AND sctr = ?''', (iso, 'shift', run, repr(st)))
    (config, statErrForm, systErrForm) = cur.fetchall()[0]
    
    print('Combining', iso, 'shift')
    
    cur.execute('''SELECT Lines.reference, lines.refRun FROM Runs JOIN Lines ON Runs.lineVar = Lines.line WHERE Runs.run = ?''', (run,))
    ref, refRun = cur.fetchall()[0]
    #each block is used to measure the isotope shift once
    for block in config:
        preVals, preErrs = extract(ref,'center',refRun,db,block[0])
        preVal, preErr, preRChi = weightedAverage(preVals, preErrs)
        preErr = applyChi(preErr, preRChi)
        
        intVals, intErrs = extract(iso,'center',run,db,block[1])
        
        postVals, postErrs = extract(ref,'center',refRun,db,block[2])
        postVal, postErr, postRChi = weightedAverage(postVals, postErrs)
        postErr = applyChi(postErr, postRChi)
        refMean = (preVal + postVal)/2
        if np.absolute(preVal-postVal) < np.max(preErr,postErr):
            errMean = np.sqrt(preErr**2+ postErr**2)
        else:
            errMean = np.absolute(preVal-postVal)
        
        shifts = intVals - refMean
        shiftErrors = np.sqrt(np.square(intErrs)+np.square(errMean))
        shiftMean = 0
        for i in shifts:
            shiftMean += i
        shiftMean = shiftMean/len(shifts)
        return (shifts, shiftErrors, shiftMean)
        
        
    
def applyChi(err, *rChi):
    '''Increases error by sqrt(rChi^2) if necessary. Works for several rChi as well'''
    return err * np.max(1, np.max(np.sqrt(rChi)))

def gaussProp(*args):
    '''Calculate sqrt of squared sum of args, as in gaussian error propagation'''
    return np.sqrt(sum(x**2 for x in args))