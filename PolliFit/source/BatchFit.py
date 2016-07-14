'''
Created on 15.05.2014

@author: hammen
'''

import sqlite3
import os
import sys
import traceback
import ast

from Measurement import MeasLoad

from DBIsotope import DBIsotope
from Spectra.Straight import Straight
from Spectra.FullSpec import FullSpec

from SPFitter import SPFitter

import MPLPlotter as plot


def batchFit(fileList, db, run='Run0', x_as_voltage=True, softw_gates_trs=None):
    '''Fit fileList with run and write results to db'''
    print("BatchFit started")
    print("Opening DB:", db)
    
    oldPath = os.getcwd()
    projectPath, dbname = os.path.split(db)
    os.chdir(projectPath)
    
    con = sqlite3.connect(dbname)
    cur = con.cursor()
    
    cur.execute('''SELECT isoVar, lineVar, scaler, track FROM Runs WHERE run = ?''', (run,))
    var = cur.fetchall()[0]
    st = (ast.literal_eval(var[2]), ast.literal_eval(var[3]))
    if softw_gates_trs is None:  # if no software gate provided check db
        try:  # check if there are software gates available in database
            cur.execute('''SELECT softwGates FROM Runs WHERE run = ?''', (run,))
            soft_var = cur.fetchall()[0]
            softw_gates_trs_db = ast.literal_eval(soft_var[0])
            if isinstance(softw_gates_trs_db, list):
                softw_gates_trs = softw_gates_trs_db
        except Exception as e:
            print('error while trying to extract the software Gates from Runs: ', e)
            print('will use gates from file')

    print("Go for", run, "with IsoVar = \"" + var[0] + "\" and LineVar = \"" + var[1] + "\"")
    
    errcount = 0
    fits = []
    for file in fileList:
        try:
            fits.append(singleFit(file, st, dbname, run, var, cur, x_as_voltage, softw_gates_trs))
        except:
            errcount += 1
            print("Error working on file", file, ":", sys.exc_info()[1])
            traceback.print_tb(sys.exc_info()[2])
            
    os.chdir(oldPath)
    con.commit()
    con.close()
    
    print("BatchFit finished,", errcount, "errors occured")


def singleFit(file, st, db, run, var, cur, x_as_voltage=True, softw_gates_trs=None):
    '''Fit st of file, using run. Save result to db and picture of spectrum to folder'''
    print('-----------------------------------')
    print("Fitting", file)
    cur.execute('''SELECT filePath FROM Files WHERE file = ?''', (file,))
    
    try:
        path = cur.fetchall()[0][0]
    except:
        raise Exception(str(file) + " not found in DB")

    meas = MeasLoad.load(path, db, x_as_voltage=x_as_voltage, softw_gates=softw_gates_trs)
    if meas.type == 'Kepco':
        print('Fitting Straight!')
        spec = Straight()
    else:
        try:
            # if the measurment is an .xml file it will have a self.seq_type
            if meas.seq_type == 'kepco':
                spec = Straight()
                spec.evaluate(meas.x[0][-1], (0, 1))
            else:
                iso = DBIsotope(db, meas.type, var[0], var[1])
                spec = FullSpec(iso)
        except:
            iso = DBIsotope(db, meas.type, var[0], var[1])
            spec = FullSpec(iso)

    fit = SPFitter(spec, meas, st)
    fit.fit()
    
    #Create and save graph
    fig = os.path.splitext(path)[0] + run + '.png'
    plot.plotFit(fit)
    plot.save(fig)
    plot.clear()
    
    result = fit.result()
    
    for r in result:
        #Only one unique result, according to PRIMARY KEY, thanks to INSERT OR REPLACE
        cur.execute('''INSERT OR REPLACE INTO FitRes (file, iso, run, rChi, pars) 
        VALUES (?, ?, ?, ?, ?)''', (file, r[0], run, fit.rchi, repr(r[1])))
        
    
    print("Finished fitting", file)
    return fit

