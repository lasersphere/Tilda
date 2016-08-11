'''
Created on 21.05.2014

@author: hammen, gorges

The Analyzer can extract() parameters from fit results, combineRes() to get weighted averages and combineShift() to calculate isotope shifts.
'''

import ast
import functools
import os
import sqlite3

import numpy as np

import MPLPlotter as plt
import Physics


def getFiles(iso, run, db):
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute('''SELECT file, pars FROM FitRes WHERE iso = ? AND run = ?''', (iso, run))
    e = cur.fetchall()
    con.close()
    
    return [f[0] for f in e]


def extract(iso, par, run, db, fileList=[], prin=True):
    '''Return a list of values of par of iso, filtered by files in fileList'''
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
    date_list = []

    for f, v, e in zip(files, vals, errs):
        cur.execute('''SELECT date FROM Files WHERE file = ?''', (f,))
        e = cur.fetchall()
        date = e[0][0]
        if date is not None:
            date_list.append(date)
        if prin:
            print(date, '\t', f, '\t', v, '\t', e)

    for f in fileList:
        if f not in files:
            print('Warning:', f, 'not found!')
    # for i in dates:
    #     print(i[0], '\t', i[1])
    con.close()
    return (vals, errs, date_list)
    
    
def weightedAverage(vals, errs):
    '''Return (weighted average, propagated error, rChi^2'''
    weights = 1 / np.square(errs)
    average = sum(vals * weights) / sum(weights)
    errorprop = np.sqrt(1 / sum(weights))  # was: 1 / sum(weights)
    if(len(vals) == 1):
        rChi = 0
    else:
        rChi = 1 / (len(vals) - 1) * sum(np.square(vals - average) * weights)
    
    return (average, errorprop, rChi)


def average(vals, errs):
    average = sum(vals) / len(vals)
    errorprop = np.sqrt(sum(np.square(errs))) / len(vals)
        
    if len(vals) == 1:
        rChi = 0
    else:
        #rChi = 0
        chiSum = 0
        for i in range(0,len(vals),1): 
            chiSum += np.square(vals[i] - average) * np.square(errs[i])
        rChi = 1 / (len(vals) - 1) * chiSum        
    return (average, errorprop, rChi)


def combineRes(iso, par, run, db, weighted = True, print_extracted=True, show_plot=False):
    '''Calculate weighted average of par using the configuration specified in the db
    :rtype : object
    '''
    print('Open DB', db)
    
    con = sqlite3.connect(db)
    cur = con.cursor()
    
    cur.execute('''INSERT OR IGNORE INTO Combined (iso, parname, run) VALUES (?, ?, ?)''', (iso, par, run))
    con.commit()
    
    cur.execute('''SELECT config, statErrForm, systErrForm FROM Combined WHERE iso = ? AND parname = ? AND run = ?''', (iso, par, run))
    (config, statErrForm, systErrForm) = cur.fetchall()[0]
    config = ast.literal_eval(config)
    
    print('Combining', iso, par)
    vals, errs, date = extract(iso, par, run, db, config, prin=print_extracted)
    
    if weighted:
        avg, err, rChi = weightedAverage(vals, errs)
    else:
        avg, err, rChi = average(vals, errs)
    print('rChi is: ', rChi, 'err is: ', err)

    statErr = eval(statErrForm)
    systErr = eval(systErrForm)
    print('statErr is: ', statErr)
    
    print('Statistical error formula:', statErrForm)
    print('Systematic error formula:', systErrForm)
    print('Combined to', iso, par, '=')
    print(str(avg) + '(' + str(statErr) + ')[' + str(systErr) + ']')
    
    cur.execute('''UPDATE Combined SET val = ?, statErr = ?, systErr = ?, rChi = ?
        WHERE iso = ? AND parname = ? AND run = ?''', (avg, statErr, systErr, rChi, iso, par, run))

    con.commit()
    con.close()
    plt.clear()
    plotdata = (date, vals, errs, avg, statErr, systErr, ('k.', 'r'))
    plt.plotAverage(*plotdata)
    combined_plots_dir = os.path.join(os.path.split(db)[0], 'combined_plots')
    if not os.path.exists(combined_plots_dir):
        os.makedirs(combined_plots_dir)
    avg_fig_name = os.path.join(combined_plots_dir, iso + '_' + run + '_' + par + '.png')
    print('saving average plot to: ', avg_fig_name)
    plt.save(avg_fig_name)
    if show_plot:
        plt.show(True)
    plt.clear()
    return (avg, statErr, systErr, plotdata)


def combineShift(iso, run, db, show_plot=False):
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
    (ref, refRun) = cur.fetchall()[0]
    #each block is used to measure the isotope shift once
    shifts = []
    shiftErrors = []
    dateIso = []
    for block in config:
        if block[0]:
            preVals, preErrs, date = extract(ref,'center',refRun,db,block[0])
            preVal, preErr, preRChi = weightedAverage(preVals, preErrs)
            preErr = applyChi(preErr, preRChi)
            preErr = np.absolute(preErr)
        else:
            preVal = 0
            preErr = 0

        intVals, intErrs, date = extract(iso,'center',run,db,block[1])
        [dateIso.append(i) for i in date]

        if block[2]:
            postVals, postErrs, date = extract(ref,'center',refRun,db,block[2])
            postVal, postErr, postRChi = weightedAverage(postVals, postErrs)
            postErr = np.absolute(applyChi(postErr, postRChi))
        else:
            postVal = 0
            postErr = 0
        if preVal == 0:
            refMean = postVal
        elif postVal == 0:
            refMean = preVal
        else:
            refMean = (preVal + postVal)/2

        if preVal == 0 or postVal == 0 or np.absolute(preVal-postVal) < np.max([preErr,postErr]):
            errMean = np.sqrt(preErr**2+ postErr**2)
        else:
            errMean = np.absolute(preVal-postVal)
        shifts.extend([x - refMean for x in intVals])
        shiftErrors.extend(np.sqrt(np.square(intErrs)+np.square(errMean)))
    val, err, rChi = weightedAverage(shifts, shiftErrors)   
    systE = functools.partial(shiftErr, iso, run, db, val)
    
    statErr = eval(statErrForm)
    systErr = eval(systErrForm)
    
    cur.execute('''UPDATE Combined SET val = ?, statErr = ?, systErr = ?, rChi = ?
        WHERE iso = ? AND parname = ? AND run = ?''', (val, statErr, systErr, rChi, iso, 'shift', run))
    con.commit()
    con.close()
    print('shifts:', shifts)
    print('shiftErrors:', shiftErrors)
    print('Mean of shifts:', val)
    plotdata = (dateIso, shifts, shiftErrors, val, statErr, systErr, ('k.', 'r'))
    plt.plotAverage(*plotdata)
    combined_plots_dir = os.path.join(os.path.split(db)[0], 'combined_plots')
    if not os.path.exists(combined_plots_dir):
        os.makedirs(combined_plots_dir)
    avg_fig_name = os.path.join(combined_plots_dir, iso + '_' + run + '_shift.png')
    print('saving average plot to: ', avg_fig_name)
    plt.save(avg_fig_name)
    if show_plot:
        plt.show(True)
    plt.clear()
    return (shifts, shiftErrors, val, statErr, systErr, rChi)
        
        
    
def applyChi(err, rChi):
    '''Increases error by sqrt(rChi^2) if necessary. Works for several rChi as well'''
    return err * np.max([1, np.sqrt(rChi)])


def gaussProp(*args):
    '''Calculate sqrt of squared sum of args, as in gaussian error propagation'''
    return np.sqrt(sum(x**2 for x in args))


def shiftErr(iso, run, db, val, accVolt_d, offset_d):
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute('''SELECT Lines.reference, lines.frequency FROM Runs JOIN Lines ON Runs.lineVar = Lines.lineVar WHERE Runs.run = ?''', (run,))
    (ref, nu0) = cur.fetchall()[0]
    cur.execute('''SELECT mass, mass_d FROM Isotopes WHERE iso = ?''', (iso,))
    (mass, mass_d) = cur.fetchall()[0]
    cur.execute('''SELECT mass, mass_d FROM Isotopes WHERE iso = ?''', (ref,))
    (massRef, massRef_d) = cur.fetchall()[0]
    deltaM = np.absolute(mass - massRef)
    cur.execute('''SELECT offset, accVolt, voltDivRatio FROM Files WHERE type = ?''', (iso,))
    (offset, accVolt, voltDivRatio) = cur.fetchall()[0]
    voltDivRatio = ast.literal_eval(voltDivRatio)
    offset = np.abs(offset)*voltDivRatio['offset']
    cur.execute('''SELECT offset FROM Files WHERE type = ?''', (ref,))
    (refOffset,) = cur.fetchall()[0]    
    accVolt = np.absolute(refOffset)*voltDivRatio['offset']+accVolt*voltDivRatio['accVolt']
   
    #cur.execute('''SELECT line FROM Files WHERE type = ?''', (ref,))
    '''
    (line,) = cur.fetchall()[0]
    if line == 'D1':
        if iso == '40_Ca':
            offset = 500
        elif iso == '44_Ca':
            offset = np.absolute(refOffset-offset)
        else:
            offset = np.absolute(refOffset-offset) + 700
    else:
        offset = np.absolute(refOffset-offset)
    print('offsetvoltage:', offset)
    '''
    fac = nu0*np.sqrt(Physics.qe*accVolt/(2*mass*Physics.u*Physics.c**2))
    print('systematic error inputs caused by error of...\n...acc Voltage:',fac*(0.5*(offset/accVolt+deltaM/mass)*(accVolt_d)),'MHz  ...offset Voltage',fac*offset*offset_d/accVolt,'MHz  ...masses:',fac*(mass_d/mass+massRef_d/massRef),'MHz')

    return fac*(np.absolute(0.5*(offset/accVolt+deltaM/mass)*(accVolt_d))+np.absolute(offset*offset_d/accVolt)+np.absolute(mass_d/mass+massRef_d/massRef))
