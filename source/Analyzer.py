'''
Created on 21.05.2014

@author: hammen, gorges
'''

import sqlite3
import ast

import numpy as np

    
def extract(iso, par, run, db, fileList = []):
    ''''''
    print('Extracting', iso, par, )
    con = sqlite3.connect(db)
    cur = con.cursor()
    
    cur.execute('''SELECT file, pars FROM FitRes WHERE iso = ? AND run = ?''', (iso, run))
    fits = cur.fetchall()
    
    if fileList:
        fits = [f for f in fits if f[0] in fileList]
        
    fitres = [eval(f[1]) for f in fits]
    files = [f[0] for f in fits]
    vals = [f[par][0] for f in fitres]
    errs = [f[par][1] for f in fitres]
    
    for f, v, e in zip(files, vals, errs):
        print(f, '\t', v, '\t', e)
    
    for f in fileList:
        if f not in files:
            print('Warning:', f, 'not found!')
    
    con.close()
    return (vals, errs)
    
    
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


def combineRes(iso, par, run, db, config = []):
    print('Open DB', db)
    
    con = sqlite3.connect(db)
    cur = con.cursor()
    
    cur.execute('''INSERT OR IGNORE INTO Combined (iso, parname, run) VALUES (?, ?, ?)''', (iso, par, run))
    con.commit()
    
    cur.execute('''SELECT config, statErrForm, systErrForm FROM Combined WHERE iso = ? AND parname = ? AND run = ?''', (iso, par, run))
    (config, statErrForm, systErrForm) = cur.fetchall()[0]
    config = ast.literal_eval(config)
    
    print('Combining', iso, par)
    vals, errs = extract(iso, par, run, db, config)
    
    val, err, rChi = weightedAverage(vals, errs)
    statErr = eval(statErrForm)
    systErr = eval(systErrForm)
    
    print('Statistical error formula:', statErrForm)
    print('Systematic error formula:', systErrForm)
    print('Combined to', iso, par, '=')
    print(str(val) + '(' + str(statErr) + ')[' + str(systErr) + ']')
    
    cur.execute('''UPDATE Combined SET val = ?, statErr = ?, systErr = ?, rChi = ?
        WHERE iso = ? AND parname = ? AND run = ?''', (val, statErr, systErr, rChi, iso, par, run))

    con.commit()
    con.close()
    

def combineShift(iso, run, db):
    '''takes an Isotope a run and a database and gives the isotopeshift to the reference!'''
    print('Open DB', db)
    
    con = sqlite3.connect(db)
    cur = con.cursor()
    
    cur.execute('''SELECT config, statErrForm, systErrForm FROM Combined WHERE iso = ? AND parname = ? AND run = ?''', (iso, 'shift', run))
    (config, statErrForm, systErrForm) = cur.fetchall()[0]
    '''config needs to have this shape: [(['dataREF1.*','dataREF2.*',...],['dataINTERESTING1.*','dataINT2.*',...],['dataREF4.*',...]), ([...],[...],[...]), ...]'''
    config = ast.literal_eval(config)
    print('Combining', iso, 'shift')
    
    cur.execute('''SELECT Lines.reference, lines.refRun FROM Runs JOIN Lines ON Runs.lineVar = Lines.lineVar WHERE Runs.run = ?''', (run,))
    ref, refRun = cur.fetchall()[0]
    #each block is used to measure the isotope shift once
    shifts = []
    shiftErrors = []
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
        shifts.extend([x - refMean for x in intVals])
        shiftErrors.extend(np.sqrt(np.square(intErrs)+np.square(errMean)))
    add = 0
    for i in shifts:
        add += i
    shiftMean = add/len(shifts)
    statErr = 0
    systErr = 0
    rChi = 0
    #print('shifts:', shifts, 'shiftErrors: ',shiftErrors, 'shiftMean: ',shiftMean)
    cur.execute('''UPDATE Combined SET val = ?, statErr = ?, systErr = ?, rChi = ?
        WHERE iso = ? AND parname = ? AND run = ?''', (shiftMean, statErr, systErr, rChi, iso, 'shift', run))
    con.commit()
    return (shifts, shiftErrors, shiftMean)
        
        
    
def applyChi(err, rChi):
    '''Increases error by sqrt(rChi^2) if necessary. Works for several rChi as well'''
    return err * np.max([1, np.sqrt(rChi)])

def gaussProp(*args):
    '''Calculate sqrt of squared sum of args, as in gaussian error propagation'''
    return np.sqrt(sum(x**2 for x in args))
